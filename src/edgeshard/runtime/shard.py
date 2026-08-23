"""ModelShard — deployed model partition with session-local KV cache.

A Shard owns:
- A subset of model layers (via ModelAdapter)
- Session-local KV caches (never leaves the Shard)
- Execution engine for prefill/decode operations

Runtime API:
    create_session()
    prefill(session_id, input_ids) -> logits or hidden_states
    decode(session_id, token_id) -> logits or hidden_states
    release_session(session_id)

Pipeline roles:
- First shard: has embedding layer, receives token IDs
- Middle shards: receive hidden states from previous shard
- Last shard: has LM head, produces logits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from edgeshard.common.errors import SessionError, ShardError
from edgeshard.common.identifiers import SessionId
from edgeshard.common.logging import get_logger
from edgeshard.runtime.adapters.base import ModelAdapter
from edgeshard.runtime.kv_cache import get_kv_cache_manager
from edgeshard.transport.base import TensorTransport

logger = get_logger(__name__)


@dataclass
class SessionState:
    """Per-session state within a Shard.

    Each Session has its own KV cache that accumulates across
    prefill and decode calls.
    """

    session_id: SessionId
    kv_cache: Any
    sequence_length: int = 0


class ModelShard:
    """Deployed model partition with session management.

    A Shard is not a Worker. One Worker may host multiple Shards.
    """

    def __init__(
        self,
        shard_id: str,
        adapter: ModelAdapter,
        transport: TensorTransport | None = None,
        is_first_shard: bool = False,
        is_last_shard: bool = False,
    ) -> None:
        """Initialize ModelShard.

        Args:
            shard_id: Unique shard identifier.
            adapter: ModelAdapter for this shard's layers.
            transport: Optional TensorTransport for inter-shard communication.
            is_first_shard: True if this shard has the embedding layer.
            is_last_shard: True if this shard has the LM head.
        """
        self._shard_id = shard_id
        self._adapter = adapter
        self._transport = transport
        self._is_first_shard = is_first_shard
        self._is_last_shard = is_last_shard
        self._sessions: dict[SessionId, SessionState] = {}

        logger.info(
            f"ModelShard {shard_id} initialized "
            f"(first={is_first_shard}, last={is_last_shard})"
        )

    def create_session(
        self,
        session_id: SessionId,
        batch_size: int = 1,
        max_seq_len: int = 4096,
    ) -> None:
        """Create a new inference session on this Shard.

        Args:
            session_id: Unique session identifier.
            batch_size: Batch size for this session.
            max_seq_len: Maximum sequence length.

        Raises:
            SessionError: If session already exists.
        """
        if session_id in self._sessions:
            raise SessionError(f"Session {session_id} already exists")

        # Get device from adapter to ensure consistency
        device = self._adapter.get_device()

        kv_cache = self._adapter.init_kv_cache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=device,
        )
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            kv_cache=kv_cache,
            sequence_length=0,
        )

        # Register with KV cache manager for memory tracking
        kv_manager = get_kv_cache_manager()
        kv_manager.register_session(session_id, kv_cache, device)

        logger.debug(f"Session {session_id} created on shard {self._shard_id}")

    async def prefill(
        self,
        session_id: SessionId,
        input_ids: torch.Tensor | None = None,
        hidden_states_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run prefill (prompt processing) for a session.

        For first shard: provide input_ids (token IDs)
        For middle/last shards: provide hidden_states_input from previous shard

        Args:
            session_id: Session identifier.
            input_ids: Input token IDs [batch, prompt_len] (first shard only).
            hidden_states_input: Hidden states [batch, seq, hidden] (other shards).

        Returns:
            If last shard: logits [batch, seq, vocab]
            Otherwise: hidden states [batch, seq, hidden] for next shard

        Raises:
            SessionError: If session does not exist.
            ShardError: If input is invalid for this shard's role.
        """
        if session_id not in self._sessions:
            raise SessionError(f"Session {session_id} does not exist")

        session = self._sessions[session_id]

        # Get input for this shard
        if self._is_first_shard:
            if input_ids is None:
                raise ShardError("First shard requires input_ids")
            hidden_states = self._adapter.embed(input_ids)
            seq_len = input_ids.shape[1]
            device = input_ids.device
        else:
            if hidden_states_input is None:
                raise ShardError("Non-first shard requires hidden_states_input")
            hidden_states = hidden_states_input
            seq_len = hidden_states_input.shape[1]
            device = hidden_states_input.device

        # Prepare position IDs
        position_ids = torch.arange(
            session.sequence_length,
            session.sequence_length + seq_len,
            device=device,
        ).unsqueeze(0).expand(hidden_states.shape[0], -1)

        logger.info(
            f"Prefill: session_seq_len={session.sequence_length}, "
            f"seq_len={seq_len}, position_ids={position_ids.tolist()}"
        )

        # Forward through layers
        hidden_states, new_kv_cache = self._adapter.forward(
            hidden_states,
            session.kv_cache,
            position_ids,
        )
        session.kv_cache = new_kv_cache
        session.sequence_length += seq_len

        logger.debug(
            f"Prefill: seq_len={seq_len}, total_seq_length={session.sequence_length}, "
            f"kv_cache_type={type(session.kv_cache).__name__}"
        )

        # If last shard, compute logits
        if self._is_last_shard:
            logits = self._adapter.compute_logits(hidden_states)
            return logits

        return hidden_states

    async def decode(
        self,
        session_id: SessionId,
        token_id: int | None = None,
        hidden_states_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one decode step for a session.

        For first shard: provide token_id (single token)
        For middle/last shards: provide hidden_states_input from previous shard

        Args:
            session_id: Session identifier.
            token_id: Single token ID (first shard only).
            hidden_states_input: Hidden states [batch, 1, hidden] (other shards).

        Returns:
            If last shard: logits [batch, 1, vocab]
            Otherwise: hidden states [batch, 1, hidden] for next shard

        Raises:
            SessionError: If session does not exist.
            ShardError: If input is invalid for this shard's role.
        """
        if session_id not in self._sessions:
            raise SessionError(f"Session {session_id} does not exist")

        session = self._sessions[session_id]

        # Get input for this shard
        if self._is_first_shard:
            if token_id is None:
                raise ShardError("First shard requires token_id")
            input_ids = torch.tensor([[token_id]], device=next(
                iter(self._adapter._layers[0].parameters())
            ).device)
            hidden_states = self._adapter.embed(input_ids)
            device = input_ids.device
        else:
            if hidden_states_input is None:
                raise ShardError("Non-first shard requires hidden_states_input")
            hidden_states = hidden_states_input
            device = hidden_states_input.device

        # Prepare position IDs (single token)
        position_ids = torch.tensor(
            [[session.sequence_length]],
            device=device,
        )

        # Forward through layers
        hidden_states, new_kv_cache = self._adapter.forward(
            hidden_states,
            session.kv_cache,
            position_ids,
        )
        session.kv_cache = new_kv_cache
        session.sequence_length += 1

        logger.debug(
            f"Decode: token_id={token_id}, position={position_ids.item()}, "
            f"total_seq_length={session.sequence_length}"
        )

        # If last shard, compute logits
        if self._is_last_shard:
            logits = self._adapter.compute_logits(hidden_states)
            return logits

        return hidden_states

    def release_session(self, session_id: SessionId) -> None:
        """Release a session and free its KV cache.

        Args:
            session_id: Session identifier.

        Raises:
            SessionError: If session does not exist.
        """
        if session_id not in self._sessions:
            raise SessionError(f"Session {session_id} does not exist")

        del self._sessions[session_id]

        # Unregister from KV cache manager
        kv_manager = get_kv_cache_manager()
        kv_manager.unregister_session(session_id)

        logger.debug(f"Session {session_id} released on shard {self._shard_id}")

    @property
    def shard_id(self) -> str:
        """Return shard identifier."""
        return self._shard_id

    @property
    def is_first_shard(self) -> bool:
        """Return True if this shard has the embedding layer."""
        return self._is_first_shard

    @property
    def is_last_shard(self) -> bool:
        """Return True if this shard has the LM head."""
        return self._is_last_shard
