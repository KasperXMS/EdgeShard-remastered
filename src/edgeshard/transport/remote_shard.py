"""RemoteModelShard — client-side proxy for inference on a remote shard.

This wraps a gRPC connection to a shard running on a remote worker.
It provides the same interface as ModelShard (create_session, prefill,
decode, release_session) but calls remote inference RPCs instead of
running the model locally.

Used by the inference client to drive distributed generation across
multiple shards on different machines.
"""

from __future__ import annotations

from typing import Any

import grpc
import torch

from edgeshard._grpc import shard_pb2, shard_pb2_grpc
from edgeshard.common.identifiers import SessionId
from edgeshard.common.logging import get_logger
from edgeshard.transport.grpc_transport import message_to_tensor, tensor_to_message

logger = get_logger(__name__)


class RemoteModelShard:
    """Client-side proxy for a remote shard's inference API.

    Provides the same interface as ModelShard so it can be used
    transparently by PipelineOrchestrator.
    """

    def __init__(
        self,
        shard_id: str,
        address: str,
        is_first_shard: bool = False,
        is_last_shard: bool = False,
    ) -> None:
        """Initialize remote shard proxy.

        Args:
            shard_id: Remote shard's ID.
            address: gRPC address (host:port) of the shard's data plane.
            is_first_shard: True if remote shard has embedding layer.
            is_last_shard: True if remote shard has LM head.
        """
        self._shard_id = shard_id
        self._address = address
        self._is_first_shard = is_first_shard
        self._is_last_shard = is_last_shard
        self._channel = grpc.aio.insecure_channel(address)
        self._stub = shard_pb2_grpc.ShardServiceStub(self._channel)

        logger.info(f"RemoteModelShard {shard_id} connected to {address}")

    @property
    def shard_id(self) -> str:
        return self._shard_id

    @property
    def is_first_shard(self) -> bool:
        return self._is_first_shard

    @property
    def is_last_shard(self) -> bool:
        return self._is_last_shard

    def create_session(
        self,
        session_id: SessionId,
        batch_size: int = 1,
        max_seq_len: int = 4096,
    ) -> None:
        """Create an inference session on the remote shard (sync)."""
        # Use synchronous gRPC for session management
        sync_channel = grpc.insecure_channel(self._address)
        sync_stub = shard_pb2_grpc.ShardServiceStub(sync_channel)
        try:
            response = sync_stub.CreateSession(
                shard_pb2.CreateSessionRequest(
                    session_id=str(session_id),
                    batch_size=batch_size,
                    max_seq_len=max_seq_len,
                )
            )
            if not response.success:
                raise RuntimeError(f"CreateSession failed: {response.message}")
            logger.debug(f"Session {session_id} created on remote shard {self._shard_id}")
        finally:
            sync_channel.close()

    async def prefill(
        self,
        session_id: SessionId,
        input_ids: torch.Tensor | None = None,
        hidden_states_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run prefill on the remote shard.

        Args:
            session_id: Session ID.
            input_ids: Token IDs [batch, seq_len] (first shard only).
            hidden_states_input: Hidden states [batch, seq, hidden] (non-first shards).

        Returns:
            Output tensor: logits (last shard) or hidden states (other shards).
        """
        request = shard_pb2.InferenceRequest(session_id=str(session_id))

        if input_ids is not None:
            request.input_ids.CopyFrom(tensor_to_message(input_ids, str(session_id)))
        if hidden_states_input is not None:
            request.hidden_states.CopyFrom(tensor_to_message(hidden_states_input, str(session_id)))

        response = await self._stub.RunPrefill(request)
        if not response.success:
            raise RuntimeError(f"RunPrefill failed on shard {self._shard_id}: {response.message}")

        # Deserialize output tensor to CPU (client doesn't need GPU for argmax)
        output = message_to_tensor(response.output, torch.device("cpu"))
        logger.debug(
            f"Prefill on {self._shard_id}: output shape={tuple(output.shape)}"
        )
        return output

    async def decode(
        self,
        session_id: SessionId,
        token_id: int | None = None,
        hidden_states_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one decode step on the remote shard.

        Args:
            session_id: Session ID.
            token_id: Single token ID (first shard only).
            hidden_states_input: Hidden states [batch, 1, hidden] (non-first shards).

        Returns:
            Output tensor: logits (last shard) or hidden states (other shards).
        """
        request = shard_pb2.InferenceRequest(session_id=str(session_id))

        if token_id is not None:
            request.token_id = token_id
        if hidden_states_input is not None:
            request.hidden_states.CopyFrom(tensor_to_message(hidden_states_input, str(session_id)))

        response = await self._stub.RunDecode(request)
        if not response.success:
            raise RuntimeError(f"RunDecode failed on shard {self._shard_id}: {response.message}")

        output = message_to_tensor(response.output, torch.device("cpu"))
        logger.debug(
            f"Decode on {self._shard_id}: output shape={tuple(output.shape)}"
        )
        return output

    def release_session(self, session_id: SessionId) -> None:
        """Release a session on the remote shard (sync)."""
        sync_channel = grpc.insecure_channel(self._address)
        sync_stub = shard_pb2_grpc.ShardServiceStub(sync_channel)
        try:
            response = sync_stub.ReleaseSession(
                shard_pb2.ReleaseSessionRequest(session_id=str(session_id))
            )
            if not response.success:
                logger.warning(f"ReleaseSession failed: {response.message}")
            logger.debug(f"Session {session_id} released on remote shard {self._shard_id}")
        finally:
            sync_channel.close()

    async def close(self) -> None:
        """Close the gRPC channel."""
        await self._channel.close()
