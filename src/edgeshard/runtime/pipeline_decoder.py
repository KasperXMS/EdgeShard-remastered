"""Pipeline decoder — distributed greedy decoding across multiple shards.

This is the high-level API for distributed text generation.
It wraps PipelineOrchestrator and provides:
- Tokenizer integration
- Autoregressive decoding loop
- Greedy token selection
- Stopping conditions

Usage:
    pipeline = PipelineOrchestrator([shard0, shard1, shard2])
    decoder = PipelineDecoder(pipeline, tokenizer)
    result = await decoder.generate("Hello world", config)
"""

from __future__ import annotations

from typing import Any

import torch

from edgeshard.common.errors import ShardError
from edgeshard.common.identifiers import SessionId
from edgeshard.common.logging import get_logger
from edgeshard.runtime.decoder import GenerationConfig, GenerationResult
from edgeshard.runtime.pipeline import PipelineOrchestrator

logger = get_logger(__name__)


class PipelineDecoder:
    """Distributed greedy decoder for multi-shard pipelines.

    This decoder orchestrates autoregressive generation across
    multiple shards, each potentially on a different machine.
    """

    def __init__(
        self,
        pipeline: PipelineOrchestrator,
        tokenizer: Any,  # transformers.PreTrainedTokenizer
    ) -> None:
        """Initialize decoder.

        Args:
            pipeline: PipelineOrchestrator managing the shards.
            tokenizer: Hugging Face tokenizer for the model.
        """
        self._pipeline = pipeline
        self._tokenizer = tokenizer
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt using distributed greedy decoding.

        Args:
            prompt: Input text prompt.
            config: Generation configuration.

        Returns:
            GenerationResult with token IDs and decoded text.
        """
        if config is None:
            config = GenerationConfig()

        # Encode prompt
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        prompt_len = input_ids.shape[1]

        # Create session across all shards
        session_id = SessionId.generate()
        max_seq_len = prompt_len + config.max_new_tokens
        self._pipeline.create_session(session_id, batch_size=1, max_seq_len=max_seq_len)

        try:
            # Prefill: process prompt through entire pipeline
            logits = await self._pipeline.prefill(session_id, input_ids=input_ids)

            logger.debug(f"Prefill logits shape: {logits.shape}")

            # Get first token from last position
            next_token_logits = logits[0, -1, :]
            next_token = self._select_token(next_token_logits, config)

            generated_tokens = [next_token]
            logger.debug(f"Generated token 1: {next_token} ({self._tokenizer.decode([next_token])})")

            # Decode loop
            for i in range(config.max_new_tokens - 1):
                # Check for EOS
                if config.eos_token_id is not None and next_token == config.eos_token_id:
                    logger.info(f"EOS reached at token {i + 1}")
                    break

                # Single-token decode through entire pipeline
                logits = await self._pipeline.decode(session_id, token_id=next_token)

                logger.debug(f"Decode step {i+2}: logits shape={logits.shape}")

                next_token_logits = logits[0, -1, :]
                next_token = self._select_token(next_token_logits, config)

                generated_tokens.append(next_token)
                logger.debug(f"Generated token {i + 2}: {next_token} ({self._tokenizer.decode([next_token])})")

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
            self._pipeline.release_session(session_id)

    def _select_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig,
    ) -> int:
        """Select next token from logits (greedy: argmax).

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
