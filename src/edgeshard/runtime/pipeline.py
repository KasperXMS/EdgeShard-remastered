"""Pipeline orchestrator — coordinates multi-shard inference.

The PipelineOrchestrator:
- Manages a list of ModelShards in pipeline order
- Routes prefill/decode through the pipeline
- Handles tensor transfers between shards via TensorTransport
- Maintains session state across shards

This is the core of distributed inference: it takes a single input
and orchestrates the forward pass across multiple shards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from edgeshard.common.errors import SessionError, ShardError
from edgeshard.common.identifiers import SessionId
from edgeshard.common.logging import get_logger
from edgeshard.runtime.shard import ModelShard
from edgeshard.transport.base import TensorTransport

logger = get_logger(__name__)


@dataclass
class ShardEndpoint:
    """A shard in the pipeline with its transport.

    Attributes:
        shard: The ModelShard instance.
        transport: TensorTransport for inter-shard communication (None for local).
        address: gRPC address if remote (e.g., "192.168.1.10:50100").
        is_local: True if this shard runs in the same process.
    """

    shard: ModelShard
    transport: TensorTransport | None
    address: str | None
    is_local: bool


class PipelineOrchestrator:
    """Orchestrates inference across multiple shards in a pipeline.

    Pipeline order: shard[0] → shard[1] → ... → shard[N-1]
    - shard[0] is first (has embedding)
    - shard[N-1] is last (has LM head)
    - Middle shards just forward hidden states

    For prefill:
        input_ids → shard[0].prefill() → hidden_states
        hidden_states → shard[1].prefill() → hidden_states
        ...
        hidden_states → shard[N-1].prefill() → logits

    For decode:
        token_id → shard[0].decode() → hidden_states
        hidden_states → shard[1].decode() → hidden_states
        ...
        hidden_states → shard[N-1].decode() → logits
    """

    def __init__(self, shards: list[ShardEndpoint]) -> None:
        """Initialize pipeline orchestrator.

        Args:
            shards: List of ShardEndpoints in pipeline order.
        """
        if not shards:
            raise ShardError("Pipeline must have at least one shard")

        self._shards = shards
        self._sessions: dict[SessionId, list[Any]] = {}  # session_id -> per-shard state

        # Validate pipeline
        if not shards[0].shard.is_first_shard:
            raise ShardError("First shard must have embedding layer")
        if not shards[-1].shard.is_last_shard:
            raise ShardError("Last shard must have LM head")

        logger.info(f"Pipeline initialized with {len(shards)} shard(s)")
        for i, ep in enumerate(shards):
            logger.info(
                f"  Shard {i}: {ep.shard.shard_id} "
                f"(local={ep.is_local}, address={ep.address})"
            )

    def create_session(
        self,
        session_id: SessionId,
        batch_size: int = 1,
        max_seq_len: int = 4096,
    ) -> None:
        """Create a session across all shards in the pipeline.

        Args:
            session_id: Unique session identifier.
            batch_size: Batch size.
            max_seq_len: Maximum sequence length.
        """
        if session_id in self._sessions:
            raise SessionError(f"Session {session_id} already exists")

        # Create session on each shard
        for endpoint in self._shards:
            endpoint.shard.create_session(
                session_id,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            )

        self._sessions[session_id] = [None] * len(self._shards)
        logger.debug(f"Session {session_id} created on {len(self._shards)} shards")

    async def prefill(
        self,
        session_id: SessionId,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run prefill through the entire pipeline.

        Args:
            session_id: Session identifier.
            input_ids: Input token IDs [batch, seq_len].

        Returns:
            Logits from the last shard [batch, seq_len, vocab_size].

        Raises:
            SessionError: If session does not exist.
        """
        if session_id not in self._sessions:
            raise SessionError(f"Session {session_id} does not exist")

        # Single-shard pipeline: just embed + forward + logits
        if len(self._shards) == 1:
            return await self._shards[0].shard.prefill(
                session_id, input_ids=input_ids
            )

        # First shard: embed + forward
        first_shard = self._shards[0]
        hidden_states = await first_shard.shard.prefill(
            session_id, input_ids=input_ids
        )

        # Middle shards: forward hidden states
        for i in range(1, len(self._shards) - 1):
            endpoint = self._shards[i]
            prev_endpoint = self._shards[i - 1]

            # Transfer hidden states if needed
            if not endpoint.is_local:
                # Send via transport
                if prev_endpoint.transport:
                    await prev_endpoint.transport.send(
                        hidden_states,
                        endpoint.shard.shard_id,
                        session_id=str(session_id),
                    )
                # Receive on target shard (handled by shard server)
                hidden_states = await endpoint.shard.prefill(
                    session_id, hidden_states_input=hidden_states
                )
            else:
                # Local transfer
                hidden_states = await endpoint.shard.prefill(
                    session_id, hidden_states_input=hidden_states
                )

        # Last shard: forward + logits
        last_shard = self._shards[-1]
        if len(self._shards) > 1:
            # Transfer to last shard if needed
            prev_endpoint = self._shards[-2]
            if not last_shard.is_local and prev_endpoint.transport:
                await prev_endpoint.transport.send(
                    hidden_states,
                    last_shard.shard.shard_id,
                    session_id=str(session_id),
                )

        logits = await last_shard.shard.prefill(
            session_id, hidden_states_input=hidden_states
        )

        return logits

    async def decode(
        self,
        session_id: SessionId,
        token_id: int,
    ) -> torch.Tensor:
        """Run one decode step through the entire pipeline.

        Args:
            session_id: Session identifier.
            token_id: Single token ID.

        Returns:
            Logits from the last shard [batch, 1, vocab_size].

        Raises:
            SessionError: If session does not exist.
        """
        if session_id not in self._sessions:
            raise SessionError(f"Session {session_id} does not exist")

        # Single-shard pipeline: just embed + forward + logits
        if len(self._shards) == 1:
            return await self._shards[0].shard.decode(
                session_id, token_id=token_id
            )

        # First shard: embed + forward
        first_shard = self._shards[0]
        hidden_states = await first_shard.shard.decode(
            session_id, token_id=token_id
        )

        # Middle shards: forward hidden states
        for i in range(1, len(self._shards) - 1):
            endpoint = self._shards[i]
            prev_endpoint = self._shards[i - 1]

            if not endpoint.is_local and prev_endpoint.transport:
                await prev_endpoint.transport.send(
                    hidden_states,
                    endpoint.shard.shard_id,
                    session_id=str(session_id),
                )

            hidden_states = await endpoint.shard.decode(
                session_id, hidden_states_input=hidden_states
            )

        # Last shard: forward + logits
        last_shard = self._shards[-1]
        if len(self._shards) > 1:
            prev_endpoint = self._shards[-2]
            if not last_shard.is_local and prev_endpoint.transport:
                await prev_endpoint.transport.send(
                    hidden_states,
                    last_shard.shard.shard_id,
                    session_id=str(session_id),
                )

        logits = await last_shard.shard.decode(
            session_id, hidden_states_input=hidden_states
        )

        return logits

    def release_session(self, session_id: SessionId) -> None:
        """Release a session across all shards.

        Args:
            session_id: Session identifier.
        """
        if session_id not in self._sessions:
            raise SessionError(f"Session {session_id} does not exist")

        for endpoint in self._shards:
            endpoint.shard.release_session(session_id)

        del self._sessions[session_id]
        logger.debug(f"Session {session_id} released from pipeline")

    async def close(self) -> None:
        """Close all transports."""
        for endpoint in self._shards:
            if endpoint.transport:
                await endpoint.transport.close()
        logger.info("Pipeline orchestrator closed")
