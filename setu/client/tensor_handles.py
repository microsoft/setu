"""
Read/Write handles for thread-safe tensor shard access.
"""

from __future__ import annotations

import torch

from setu._commons.datatypes import (
    TensorShard,
)
from setu._commons.datatypes import TensorShardReadHandle as NativeReadHandle
from setu._commons.datatypes import TensorShardWriteHandle as NativeWriteHandle


class TensorReadHandle:
    """Context manager for read access to tensor shard device memory."""

    def __init__(self, shard: TensorShard) -> None:
        """
        Initialize read handle.

        Args:
            shard: TensorShard with tensor data and lock file path
        """
        self._shard = shard
        self._handle: NativeReadHandle | None = None

    def __enter__(self) -> torch.Tensor:
        """
        Acquire read lock and return tensor view.

        Returns:
            PyTorch tensor view of the shard's device memory
        """
        self._handle = NativeReadHandle(self._shard)
        return self._shard.tensor

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release read lock."""
        self._handle = None


class TensorWriteHandle:
    """Context manager for write access to tensor shard device memory."""

    def __init__(self, shard: TensorShard) -> None:
        """
        Initialize write handle.

        Args:
            shard: TensorShard with tensor data and lock file path
        """
        self._shard = shard
        self._handle: NativeWriteHandle | None = None

    def __enter__(self) -> torch.Tensor:
        """
        Acquire write lock and return tensor view.

        Returns:
            PyTorch tensor view of the shard's device memory
        """
        self._handle = NativeWriteHandle(self._shard)
        return self._shard.tensor

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release write lock."""
        self._handle = None
