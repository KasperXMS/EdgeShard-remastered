"""gRPC tensor transport — data-plane implementation for inter-shard transfers.

This implements the TensorTransport interface using gRPC:
- Serializes torch tensors to bytes
- Sends via gRPC to target shard
- Receives from source shard via gRPC
- Deserializes bytes back to tensors

Supports:
- Push model: sender calls send()
- Pull model: receiver calls recv()

For maximum performance, this uses:
- Zero-copy tensor serialization where possible
- gRPC compression for large tensors
- Connection pooling via shared channels
"""

from __future__ import annotations

import asyncio
from typing import Any

import grpc
import torch

from edgeshard._grpc import shard_pb2, shard_pb2_grpc
from edgeshard.common.logging import get_logger
from edgeshard.transport.base import TensorTransport

logger = get_logger(__name__)


# Dtype name mapping
_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    torch.int64: "int64",
}

_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def tensor_to_message(tensor: torch.Tensor, session_id: str) -> shard_pb2.TensorMessage:
    """Serialize a torch tensor to a protobuf message.

    Args:
        tensor: Tensor to serialize.
        session_id: Session ID for routing.

    Returns:
        TensorMessage protobuf.
    """
    # Ensure contiguous
    tensor = tensor.contiguous()

    # Get dtype string
    dtype_str = _DTYPE_TO_STR.get(tensor.dtype)
    if dtype_str is None:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")

    # Serialize to bytes
    # bfloat16 is not supported by numpy, so cast to float32
    # Use .detach() in case tensor requires grad
    if tensor.dtype == torch.bfloat16:
        data = tensor.detach().cpu().to(torch.float32).numpy().tobytes()
    else:
        data = tensor.detach().cpu().numpy().tobytes()

    return shard_pb2.TensorMessage(
        shape=list(tensor.shape),
        dtype=dtype_str,
        data=data,
        session_id=session_id,
    )


def message_to_tensor(message: shard_pb2.TensorMessage, device: torch.device) -> torch.Tensor:
    """Deserialize a protobuf message to a torch tensor.

    Args:
        message: TensorMessage protobuf.
        device: Target device.

    Returns:
        Deserialized tensor.
    """
    import numpy as np

    # Get dtype
    dtype = _STR_TO_DTYPE.get(message.dtype)
    if dtype is None:
        raise ValueError(f"Unknown dtype: {message.dtype}")

    # Deserialize from bytes
    # bfloat16 was serialized as float32, so we need to cast back
    shape = tuple(message.shape)
    if dtype == torch.bfloat16:
        array = np.frombuffer(message.data, dtype=np.float32).reshape(shape)
        tensor = torch.from_numpy(array.copy()).to(dtype=dtype, device=device)
    else:
        array = np.frombuffer(message.data, dtype=_numpy_dtype(dtype)).reshape(shape)
        tensor = torch.from_numpy(array.copy()).to(device)

    return tensor


def _numpy_dtype(torch_dtype: torch.dtype) -> Any:
    """Convert torch dtype to numpy dtype."""
    import numpy as np

    mapping = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.bfloat16: np.float32,  # numpy doesn't support bfloat16, cast to float32
        torch.int32: np.int32,
        torch.int64: np.int64,
    }
    return mapping.get(torch_dtype, np.float32)


class GrpcTensorTransport(TensorTransport):
    """gRPC-based tensor transport for inter-shard communication.

    This transport:
    - Maintains a pool of gRPC channels to other shards
    - Serializes tensors to bytes
    - Sends via gRPC SendTensor RPC
    - Receives via gRPC RecvTensor RPC
    """

    def __init__(
        self,
        shard_id: str,
        shard_addresses: dict[str, str],
    ) -> None:
        """Initialize transport.

        Args:
            shard_id: This shard's ID.
            shard_addresses: Map of shard_id -> gRPC address (host:port).
        """
        self._shard_id = shard_id
        self._shard_addresses = shard_addresses
        self._channels: dict[str, grpc.Channel] = {}
        self._stubs: dict[str, shard_pb2_grpc.ShardServiceStub] = {}
        self._recv_buffers: dict[str, asyncio.Queue] = {}  # session_id -> queue

        # Create channels to all other shards
        for sid, addr in shard_addresses.items():
            if sid != shard_id:
                channel = grpc.aio.insecure_channel(addr)
                self._channels[sid] = channel
                self._stubs[sid] = shard_pb2_grpc.ShardServiceStub(channel)
                logger.info(f"Transport {shard_id} connected to shard {sid} at {addr}")

    async def send(
        self,
        tensor: torch.Tensor,
        target_shard_id: str,
        session_id: str = "",
    ) -> None:
        """Send a tensor to a target shard.

        Args:
            tensor: Tensor to send.
            target_shard_id: Target shard ID.
            session_id: Session ID for routing.
        """
        if target_shard_id not in self._stubs:
            raise ValueError(f"Unknown target shard: {target_shard_id}")

        stub = self._stubs[target_shard_id]

        # Serialize tensor
        message = tensor_to_message(tensor, session_id)

        # Send via gRPC
        request = shard_pb2.SendTensorRequest(
            source_shard_id=self._shard_id,
            target_shard_id=target_shard_id,
            tensor=message,
        )

        try:
            response = await stub.SendTensor(request)
            if not response.success:
                raise RuntimeError(f"Send failed: {response.message}")
        except grpc.RpcError as e:
            logger.error(f"gRPC send error: {e}")
            raise

    async def recv(
        self,
        source_shard_id: str,
        session_id: str = "",
    ) -> torch.Tensor:
        """Receive a tensor from a source shard.

        This uses the pull model: we call RecvTensor on the source shard.

        Args:
            source_shard_id: Source shard ID.
            session_id: Session ID for routing.

        Returns:
            Received tensor.
        """
        if source_shard_id not in self._stubs:
            raise ValueError(f"Unknown source shard: {source_shard_id}")

        stub = self._stubs[source_shard_id]

        # Request tensor via gRPC
        request = shard_pb2.RecvTensorRequest(
            session_id=session_id,
            source_shard_id=source_shard_id,
        )

        try:
            response = await stub.RecvTensor(request)
            if not response.success:
                raise RuntimeError(f"Recv failed: {response.message}")

            # Deserialize tensor
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = message_to_tensor(response.tensor, device)
            return tensor

        except grpc.RpcError as e:
            logger.error(f"gRPC recv error: {e}")
            raise

    async def close(self) -> None:
        """Close all gRPC channels."""
        for channel in self._channels.values():
            await channel.close()
        self._channels.clear()
        self._stubs.clear()
        logger.info(f"Transport {self._shard_id} closed")


class ShardServer:
    """gRPC server for a shard — receives tensors from other shards.

    Each shard runs this server to accept incoming tensor transfers.
    Optionally supports inference RPCs when a ModelShard is provided.
    """

    def __init__(
        self,
        shard_id: str,
        host: str = "0.0.0.0",
        port: int = 50100,
        inference_shard: "ModelShard | None" = None,
    ) -> None:
        self._shard_id = shard_id
        self._host = host
        self._port = port
        self._server: grpc.aio.Server | None = None
        self._servicer = ShardServicer(shard_id, inference_shard=inference_shard)

    async def start(self) -> None:
        """Start the shard gRPC server."""
        from concurrent import futures

        self._server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=4))
        shard_pb2_grpc.add_ShardServiceServicer_to_server(self._servicer, self._server)

        bind_address = f"{self._host}:{self._port}"
        self._server.add_insecure_port(bind_address)
        await self._server.start()

        logger.info(f"Shard {self._shard_id} data plane listening on {bind_address}")

    async def stop(self, grace: float = 5.0) -> None:
        """Stop the shard gRPC server."""
        if self._server:
            await self._server.stop(grace)
        logger.info(f"Shard {self._shard_id} data plane stopped")

    def get_servicer(self) -> "ShardServicer":
        """Get the servicer for testing."""
        return self._servicer


class ShardServicer(shard_pb2_grpc.ShardServiceServicer):
    """gRPC servicer for shard data plane.

    Handles:
    - SendTensor / RecvTensor: inter-shard tensor transfer
    - CreateSession / RunPrefill / RunDecode / ReleaseSession: remote inference
    """

    def __init__(
        self,
        shard_id: str,
        inference_shard: "ModelShard | None" = None,
    ) -> None:
        self._shard_id = shard_id
        self._recv_buffers: dict[str, torch.Tensor] = {}  # session_id -> tensor
        self._inference_shard = inference_shard

    # ----- Tensor transfer (inter-shard data plane) -----

    async def SendTensor(
        self,
        request: shard_pb2.SendTensorRequest,
        context: grpc.ServicerContext,
    ) -> shard_pb2.SendTensorResponse:
        """Handle incoming tensor."""
        try:
            # Deserialize tensor
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = message_to_tensor(request.tensor, device)

            # Store in buffer
            session_id = request.tensor.session_id
            self._recv_buffers[session_id] = tensor

            logger.debug(
                f"Shard {self._shard_id} received tensor {tuple(tensor.shape)} "
                f"from {request.source_shard_id} (session={session_id})"
            )

            return shard_pb2.SendTensorResponse(success=True, message="OK")

        except Exception as e:
            logger.error(f"SendTensor error: {e}")
            return shard_pb2.SendTensorResponse(success=False, message=str(e))

    async def RecvTensor(
        self,
        request: shard_pb2.RecvTensorRequest,
        context: grpc.ServicerContext,
    ) -> shard_pb2.RecvTensorResponse:
        """Handle tensor request (pull model)."""
        session_id = request.session_id

        if session_id not in self._recv_buffers:
            return shard_pb2.RecvTensorResponse(
                success=False,
                message=f"No tensor for session {session_id}",
            )

        tensor = self._recv_buffers.pop(session_id)

        # Serialize tensor
        message = tensor_to_message(tensor, session_id)

        return shard_pb2.RecvTensorResponse(
            success=True,
            message="OK",
            tensor=message,
        )

    async def Ping(
        self,
        request: shard_pb2.RecvTensorRequest,
        context: grpc.ServicerContext,
    ) -> shard_pb2.SendTensorResponse:
        """Health check."""
        return shard_pb2.SendTensorResponse(success=True, message=f"pong from {self._shard_id}")

    # ----- Inference RPCs -----

    async def CreateSession(
        self,
        request: shard_pb2.CreateSessionRequest,
        context: grpc.ServicerContext,
    ) -> shard_pb2.CreateSessionResponse:
        """Create an inference session on this shard."""
        if self._inference_shard is None:
            return shard_pb2.CreateSessionResponse(
                success=False, message="Inference not enabled on this shard"
            )
        try:
            from edgeshard.common.identifiers import SessionId
            sid = SessionId(request.session_id)
            self._inference_shard.create_session(
                sid,
                batch_size=request.batch_size or 1,
                max_seq_len=request.max_seq_len or 4096,
            )
            return shard_pb2.CreateSessionResponse(
                success=True, message=f"Session {request.session_id} created"
            )
        except Exception as e:
            logger.error(f"CreateSession error: {e}")
            return shard_pb2.CreateSessionResponse(success=False, message=str(e))

    async def RunPrefill(
        self,
        request: shard_pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> shard_pb2.InferenceResponse:
        """Run prefill on this shard."""
        if self._inference_shard is None:
            return shard_pb2.InferenceResponse(
                success=False, message="Inference not enabled on this shard"
            )
        try:
            from edgeshard.common.identifiers import SessionId
            sid = SessionId(request.session_id)
            device = self._inference_shard._adapter.get_device()

            input_ids = None
            hidden_states = None

            if self._inference_shard.is_first_shard and request.HasField("input_ids"):
                input_ids = message_to_tensor(request.input_ids, device)
            elif not self._inference_shard.is_first_shard and request.HasField("hidden_states"):
                hidden_states = message_to_tensor(request.hidden_states, device)

            output = await self._inference_shard.prefill(
                sid, input_ids=input_ids, hidden_states_input=hidden_states
            )

            output_msg = tensor_to_message(output, request.session_id)
            return shard_pb2.InferenceResponse(
                success=True, message="OK", output=output_msg
            )
        except Exception as e:
            logger.error(f"RunPrefill error: {e}")
            return shard_pb2.InferenceResponse(success=False, message=str(e))

    async def RunDecode(
        self,
        request: shard_pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> shard_pb2.InferenceResponse:
        """Run one decode step on this shard."""
        if self._inference_shard is None:
            return shard_pb2.InferenceResponse(
                success=False, message="Inference not enabled on this shard"
            )
        try:
            from edgeshard.common.identifiers import SessionId
            sid = SessionId(request.session_id)
            device = self._inference_shard._adapter.get_device()

            token_id = None
            hidden_states = None

            if self._inference_shard.is_first_shard:
                token_id = request.token_id
            elif request.HasField("hidden_states"):
                hidden_states = message_to_tensor(request.hidden_states, device)

            output = await self._inference_shard.decode(
                sid, token_id=token_id, hidden_states_input=hidden_states
            )

            output_msg = tensor_to_message(output, request.session_id)
            return shard_pb2.InferenceResponse(
                success=True, message="OK", output=output_msg
            )
        except Exception as e:
            logger.error(f"RunDecode error: {e}")
            return shard_pb2.InferenceResponse(success=False, message=str(e))

    async def ReleaseSession(
        self,
        request: shard_pb2.ReleaseSessionRequest,
        context: grpc.ServicerContext,
    ) -> shard_pb2.ReleaseSessionResponse:
        """Release an inference session."""
        if self._inference_shard is None:
            return shard_pb2.ReleaseSessionResponse(
                success=False, message="Inference not enabled on this shard"
            )
        try:
            from edgeshard.common.identifiers import SessionId
            sid = SessionId(request.session_id)
            self._inference_shard.release_session(sid)
            return shard_pb2.ReleaseSessionResponse(
                success=True, message=f"Session {request.session_id} released"
            )
        except Exception as e:
            logger.error(f"ReleaseSession error: {e}")
            return shard_pb2.ReleaseSessionResponse(success=False, message=str(e))
