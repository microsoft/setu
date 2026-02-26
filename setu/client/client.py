"""
Setu client API for tensor operations.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from setu._client import Client as Client_C
from setu._commons.datatypes import (
    CopySpec,
    TensorShard,
    TensorShardRef,
    TensorShardSpec,
)
from setu.client.tensor_handles import TensorReadHandle, TensorWriteHandle
from setu.client.tensor_selection import TensorSelection
from setu.core.types import CopyOperationId, TensorName

logger = logging.getLogger(__name__)


class Client:
    """
    Client interface for creating and managing tensor shards in the Setu system.

    The Client provides the primary interface for:
    - Registering tensor shards with the Setu system
    - Accessing tensor data through read/write context managers
    - Copying data between tensor selections
    - Creating tensor selections for operations

    Example:
        >>> client = Client("tcp://localhost:5555")
        >>> shard_ref = client.register_tensor_shard(spec)
        >>> with client.write(shard_ref) as tensor:
        ...     tensor.fill_(1.0)
    """

    def __init__(self, endpoint: str) -> None:
        """
        Initialize the Setu client and connect to a NodeAgent.

        Args:
            endpoint: ZMQ endpoint for the NodeAgent (e.g., "tcp://localhost:5555")

        Raises:
            RuntimeError: If connection to the endpoint fails
        """
        self._client = Client_C()
        self._client.connect(endpoint)
        self._endpoint = endpoint
        self._tensor_shard_cache: Dict[uuid.UUID, TensorShard] = {}
        logger.debug("Client connected to %s", endpoint)

    @property
    def endpoint(self) -> str:
        """Get the endpoint this client is connected to."""
        return self._endpoint

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected to a NodeAgent."""
        return self._client.is_connected()

    def disconnect(self) -> None:
        """
        Disconnect from the NodeAgent.

        This deregisters all owned shards (notifying both the NodeAgent and
        Coordinator) and closes the connection. Call this explicitly before
        the client goes out of scope to ensure clean shutdown.
        """
        if self._client.is_connected():
            self._client.disconnect()
            self._tensor_shard_cache.clear()
            logger.debug("Client disconnected from %s", self._endpoint)

    def register_tensor_shard(self, spec: TensorShardSpec) -> Optional[TensorShardRef]:
        """
        Register a tensor shard with the specified dimensions and properties.

        The shard will be allocated on the NodeAgent and tracked by the
        Coordinator. The returned TensorShardRef can be used for subsequent
        read/write operations.

        Args:
            spec: TensorShardSpec defining the tensor name, dimensions, dtype,
                  and device

        Returns:
            TensorShardRef if registration succeeds, None if it fails (e.g.,
            overlapping shards, dtype mismatch, dimension mismatch)

        Example:
            >>> from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
            >>> device = Device(torch_device=torch.device("cuda:0"))
            >>> dims = [
            ...     TensorDimSpec("rows", 1024, 0, 512),
            ...     TensorDimSpec("cols", 768, 0, 768),
            ... ]
            >>> spec = TensorShardSpec(
            ...     name="my_tensor",
            ...     dims=dims,
            ...     dtype=torch.float32,
            ...     device=device,
            ... )
            >>> shard_ref = client.register_tensor_shard(spec)
        """
        shard_ref = self._client.register_tensor_shard(spec)

        if shard_ref is not None:
            logger.debug("Registered tensor shard: %s", spec.name)

        return shard_ref

    def select(self, name: TensorName) -> TensorSelection:
        """
        Create a tensor selection covering only indices owned by this client's shards.

        The selection includes only the indices that this client has registered
        shards for. Use `.where()` to narrow the selection further.

        Args:
            name: Name of the tensor to create selection for

        Returns:
            TensorSelection covering only owned indices

        Raises:
            RuntimeError: If the tensor name is not known or selection fails

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
        native_selection = self._client.select(name)
        return TensorSelection(native_selection)

    def copy(
        self, src: TensorSelection, dst: TensorSelection
    ) -> Optional[CopyOperationId]:
        """
        Copy data from source selection to destination selection.

        Args:
            src: Source tensor selection
            dst: Destination tensor selection
            blocking: If True, wait for copy to complete before returning.
                      If False, return immediately with a CopyOperationId
                      that can be used with wait().

        Returns:
            None if blocking=True, CopyOperationId if blocking=False

        Raises:
            ValueError: If selections are incompatible (different dimensions/sizes)
            RuntimeError: If copy operation fails

        Example:
            >>> src = client.select("kv_cache_src").where("page", [1, 2, 3])
            >>> dst = client.select("kv_cache_dst").where("page", [4, 5, 6])
            >>> client.copy(src, dst)  # Blocking copy
            >>>
            >>> # Or non-blocking:
            >>> op_id = client.copy(src, dst, blocking=False)
            >>> # ... do other work ...
            >>> client.wait(op_id)
        """
        copy_spec = CopySpec(src.name, dst.name, src.native, dst.native)

        copy_op_id = self._client.submit_copy(copy_spec)

        if copy_op_id is None:
            raise RuntimeError("Copy operation submission failed")

        logger.debug(
            "Submitted copy operation %d: %s -> %s", copy_op_id, src.name, dst.name
        )
        return copy_op_id

    def wait(self, copy_op_id: CopyOperationId) -> None:
        """
        Wait for a copy operation to complete.

        Args:
            copy_op_id: The copy operation ID returned by copy(blocking=False)

        Example:
            >>> op_id = client.copy(src, dst, blocking=False)
            >>> # ... do other work ...
            >>> client.wait(op_id)
        """
        self._client.wait_for_copy(copy_op_id)
        logger.debug("Copy operation %d completed", copy_op_id)

    def _get_tensor_shard(self, shard_ref: TensorShardRef) -> TensorShard:
        """Get a cached TensorShard, or fetch and cache on first access.

        Args:
            shard_ref: TensorShardRef to look up

        Returns:
            TensorShard with rebuilt tensor, metadata, and lock_base_dir
        """
        shard_id = shard_ref.shard_id
        cached = self._tensor_shard_cache.get(shard_id)
        if cached is not None:
            return cached

        tensor_ipc_spec, metadata, lock_base_dir = self._client.get_tensor_handle(
            shard_ref
        )
        tensor = rebuild_cuda_tensor(
            **tensor_ipc_spec.to_dict(),
            tensor_cls=torch.Tensor,
            storage_cls=torch.storage.UntypedStorage,
        )
        shard = TensorShard(
            metadata=metadata,
            tensor=tensor,
            lock_base_dir=lock_base_dir,
        )
        self._tensor_shard_cache[shard_id] = shard
        return shard

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
            ...     data = tensor[0, :, :].clone()
            ...     print(tensor.sum())
        """
        shard = self._get_tensor_shard(shard_ref)
        with TensorReadHandle(shard) as tensor:
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
            ...     tensor.fill_(1.0)
            ...     tensor[0, :, :] = some_data
        """
        shard = self._get_tensor_shard(shard_ref)
        with TensorWriteHandle(shard) as tensor:
            yield tensor
