"""
Read/Write handles for thread-safe tensor shard access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from setu._commons.datatypes import TensorShard

if TYPE_CHECKING:
    from setu.client.client import Client


class TensorReadHandle:
    """Context manager for read access to tensor shard device memory."""

    def __init__(self, client: Client, shard: TensorShard) -> None:
        """
        Initialize read handle.

        Args:
            client: Client instance for accessing tensor operations
            shard: TensorShard to acquire read access for
        """
        self._client = client
        self._shard = shard
        self._tensor: torch.Tensor | None = None

    def __enter__(self) -> torch.Tensor:
        """
        Acquire read lock and return tensor view.

        Returns:
            PyTorch tensor view of the shard's device memory
        """
        # TODO: Acquire read lock on device_ptr via C++ native handle
        # TODO: Create torch.Tensor view from device_ptr
        raise NotImplementedError("ReadHandle.__enter__ not yet implemented")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release read lock."""
        # TODO: Release read lock on device_ptr via C++ native handle
        raise NotImplementedError("ReadHandle.__exit__ not yet implemented")


class TensorWriteHandle:
    """Context manager for write access to tensor shard device memory."""

    def __init__(self, client: Client, shard: TensorShard) -> None:
        """
        Initialize write handle.

        Args:
            client: Client instance for accessing tensor operations
            shard: TensorShard to acquire write access for
        """
        self._client = client
        self._shard = shard
        self._tensor: torch.Tensor | None = None

    def __enter__(self) -> torch.Tensor:
        """
        Acquire write lock and return tensor view.

        Returns:
            PyTorch tensor view of the shard's device memory
        """
        # TODO: Acquire write lock on device_ptr via C++ native handle
        # TODO: Create torch.Tensor view from device_ptr
        raise NotImplementedError("WriteHandle.__enter__ not yet implemented")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release write lock."""
        # TODO: Release write lock on device_ptr via C++ native handle
        raise NotImplementedError("WriteHandle.__exit__ not yet implemented")
