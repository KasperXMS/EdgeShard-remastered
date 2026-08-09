"""TensorTransport ABC — defines the tensor data-plane interface.

The transport layer handles hidden-state transfers between Shards.
Master is never in this path. Implementations may use gRPC, NCCL,
RDMA, or any other mechanism.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class TensorTransport(ABC):
    """Abstract interface for inter-Shard tensor transfers.

    The transport is point-to-point: a sender Shard sends a tensor
    to a receiver Shard. The Master does not participate.
    """

    @abstractmethod
    async def send(
        self,
        tensor: torch.Tensor,
        target_shard_id: str,
        session_id: str = "",
    ) -> None:
        """Send a tensor to a target Shard.

        Args:
            tensor: Hidden states to send [batch, seq, hidden].
            target_shard_id: Identifier of the receiving Shard.
            session_id: Session ID for routing (optional).
        """

    @abstractmethod
    async def recv(
        self,
        source_shard_id: str,
        session_id: str = "",
    ) -> torch.Tensor:
        """Receive a tensor from a source Shard.

        Args:
            source_shard_id: Identifier of the sending Shard.
            session_id: Session ID for routing (optional).

        Returns:
            Received hidden states [batch, seq, hidden].
        """

    @abstractmethod
    async def close(self) -> None:
        """Release transport resources."""
