"""
Setu client API for tensor operations.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List

import torch

from setu._commons.datatypes import TensorDim, TensorShardRef
from setu.client.tensor_handles import TensorReadHandle, TensorWriteHandle
from setu.client.tensor_selection import TensorSelection
from setu.core.types import TensorName


class Client:
    """
    Client interface for creating and managing tensor shards in the Setu system.
    """

    def __init__(self) -> None:
        """Initialize the Setu client."""
        self._shards: Dict[TensorName, TensorShardRef] = {}

    def create_tensor_shard(
        self,
        name: TensorName,
        dims: List[TensorDim],
        dtype: torch.dtype,
        device: str,
    ) -> TensorShardRef:
        """
        Create a tensor shard with the specified dimensions and properties.

        Args:
            name: Fully qualified name for the tensor shard in the format
                  "replica:X/worker:Y/task:Z/tensor_name"
            dims: List of TensorDim objects defining each dimension with its name
                  and size
            dtype: PyTorch data type for the tensor (e.g., torch.bfloat16, torch.float16)
            device: Device string in PyTorch format (e.g., "cuda:0", "cpu")

        Returns:
            TensorShardRef object - a reference to the created shard

        Example:
            >>> client = Client()
            >>> from setu._commons.datatypes import TensorDim
            >>> shard_ref = client.create_tensor_shard(
            ...     name="replica:0/worker:0/task:0/t",
            ...     dims=[
            ...         TensorDim("a", 2048),
            ...         TensorDim("b", 128),
            ...         TensorDim("c", 256)
            ...     ],
            ...     dtype=torch.bfloat16,
            ...     device="cuda:0"
            ... )
        """
        # TODO: Implement tensor shard creation
        raise NotImplementedError("create_tensor_shard not yet implemented")

    def select(self, name: TensorName) -> TensorSelection:
        """
        Create a tensor selection for the given tensor (initially selects all indices).

        Args:
            name: Fully qualified tensor name

        Returns:
            TensorSelection with all indices selected (use .where() to narrow)

        Example:
            >>> # Using different indexing styles
            >>> src = client.select("prefill_replica:0/layer:0/kv_cache") \\
            ...     .where("page", [0, 1, 2]) \\
            ...     .where("seq", slice(0, 32))
            >>>
            >>> # Or using sets
            >>> src = client.select("tensor_name") \\
            ...     .where("page", {0, 1, 2}) \\
            ...     .where("head", 5)  # Single index
        """
        # TODO: Get dims from metadata registry
        # For now this will need to be implemented when metadata tracking is added
        raise NotImplementedError(
            "select not yet implemented - needs metadata tracking"
        )

    def copy(self, src: TensorSelection, dst: TensorSelection) -> None:
        """
        Copy data from source selection to destination selection.

        Args:
            src: Source tensor selection
            dst: Destination tensor selection

        Raises:
            ValueError: If selections are incompatible or copy would fail

        Example:
            >>> src = client.select("kv_cache_src").where(page=[1,2,3]).build()
            >>> dst = client.select("kv_cache_dst").where(page=[4,5,6]).build()
            >>> client.copy(src, dst)
        """
        # TODO: Implement tensor copy operation
        raise NotImplementedError("copy not yet implemented")

    @contextmanager
    def read(self, shard_ref: TensorShardRef) -> Iterator[torch.Tensor]:
        """
        Context manager for read access to tensor shard.

        Args:
            shard_ref: TensorShardRef to read from

        Yields:
            PyTorch tensor view with read access

        Example:
            >>> with client.read(shard_ref) as tensor:
            ...     data = tensor[0, :, :]
        """
        handle = TensorReadHandle(self, shard_ref)
        with handle as tensor:
            yield tensor

    @contextmanager
    def write(self, shard_ref: TensorShardRef) -> Iterator[torch.Tensor]:
        """
        Context manager for write access to tensor shard.

        Args:
            shard_ref: TensorShardRef to write to

        Yields:
            PyTorch tensor view with write access

        Example:
            >>> with client.write(shard_ref) as tensor:
            ...     tensor[0, :, :] = new_data
        """
        handle = TensorWriteHandle(self, shard_ref)
        with handle as tensor:
            yield tensor
