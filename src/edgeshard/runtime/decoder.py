"""Greedy decoder — autoregressive generation with greedy token selection.

This module provides a high-level API for text generation using a ModelShard.
It handles:
- Tokenizer integration
- Prompt encoding
- Autoregressive decoding loop
- Greedy token selection (argmax)
- Stopping conditions (EOS, max_length)

For distributed inference (multi-shard), this would orchestrate the pipeline.
For M2, we focus on single-shard correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from edgeshard.common.errors import SessionError, ShardError
from edgeshard.common.identifiers import SessionId
from edgeshard.common.logging import get_logger
from edgeshard.runtime.shard import ModelShard

logger = get_logger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    eos_token_id: int | None = None
    pad_token_id: int | None = None


@dataclass
class GenerationResult:
    """Result of text generation."""

    token_ids: list[int]
    text: str
    num_tokens: int
    finished: bool  # True if stopped by EOS, False if stopped by max_length


class GreedyDecoder:
    """Greedy autoregressive decoder for a single ModelShard.

    This decoder assumes the shard is both first and last (complete model).
    For multi-shard pipelines, a different orchestrator is needed.
    """

    def __init__(
        self,
        shard: ModelShard,
        tokenizer: Any,  # transformers.PreTrainedTokenizer
    ) -> None:
        """Initialize decoder.

        Args:
            shard: ModelShard with is_first=True and is_last=True.
            tokenizer: Hugging Face tokenizer for the model.
        """
        if not (shard.is_first_shard and shard.is_last_shard):
            raise ShardError(
                "GreedyDecoder requires a complete shard (first and last). "
                "For multi-shard pipelines, use PipelineDecoder."
            )

        self._shard = shard
        self._tokenizer = tokenizer
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt using greedy decoding.

        Args:
            prompt: Input text prompt.
            config: Generation configuration (defaults to greedy).

        Returns:
            GenerationResult with token IDs and decoded text.
        """
        if config is None:
            config = GenerationConfig()

        # Encode prompt
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        prompt_len = input_ids.shape[1]

        # Create session
        session_id = SessionId.generate()
        max_seq_len = prompt_len + config.max_new_tokens
        self._shard.create_session(session_id, batch_size=1, max_seq_len=max_seq_len)

        try:
            # Prefill: process prompt
            logits = await self._shard.prefill(session_id, input_ids=input_ids)

            # Get first token from last position
            next_token_logits = logits[0, -1, :]
            next_token = self._select_token(next_token_logits, config)

            generated_tokens = [next_token]
            logger.debug(f"Generated token 1: {next_token}")

            # Decode loop
            for i in range(config.max_new_tokens - 1):
                # Check for EOS
                if config.eos_token_id is not None and next_token == config.eos_token_id:
                    logger.info(f"EOS reached at token {i + 1}")
                    break

                # Single-token decode
                logits = await self._shard.decode(session_id, token_id=next_token)
                next_token_logits = logits[0, -1, :]
                next_token = self._select_token(next_token_logits, config)

                generated_tokens.append(next_token)
                logger.debug(f"Generated token {i + 2}: {next_token}")

            # Decode to text
            generated_text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            finished = (
                config.eos_token_id is not None
                and generated_tokens[-1] == config.eos_token_id
            )

            return GenerationResult(
                token_ids=generated_tokens,
                text=prompt + generated_text,
                num_tokens=len(generated_tokens),
                finished=finished,
            )

        finally:
            # Always release session
            self._shard.release_session(session_id)

    def _select_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig,
    ) -> int:
        """Select next token from logits.

        For greedy decoding (M2), this is just argmax.
        Future: support sampling with temperature/top_k/top_p.

        Args:
            logits: Token logits [vocab_size].
            config: Generation config.

        Returns:
            Selected token ID.
        """
        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # Greedy: argmax
        next_token = torch.argmax(logits, dim=-1).item()
        return int(next_token)
