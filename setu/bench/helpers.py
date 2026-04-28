"""Helper functions for building shards and copy specs."""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from setu._commons.datatypes import (
    TensorDim,
    TensorDimSpec,
    TensorShardSpec,
)
from setu.cluster.mesh import Mesh, PartitionSpec


@dataclass(frozen=True)
class ShardedTensor:
    """A global tensor with its mesh-based sharding description.

    Wraps the tensor metadata (name, dims, dtype) together with the Mesh and
    PartitionSpec that describe how it is distributed.  The ``.shards``
    property materialises the concrete ``TensorShardSpec`` list on demand.
    """

    name: str
    dims: List[TensorDim]
    mesh: Mesh
    partition: PartitionSpec
    dtype: torch.dtype = torch.float32

    @property
    def shards(self) -> List[TensorShardSpec]:
        return shard_tensor(self.name, self.dims, self.mesh, self.partition, self.dtype)


def shard_tensor(
    name: str,
    dims: List[TensorDim],
    mesh: Mesh,
    partition: PartitionSpec,
    dtype: torch.dtype = torch.float32,
) -> List[TensorShardSpec]:
    """Produce one TensorShardSpec per mesh position from a Mesh + PartitionSpec.

    Mesh axes that are not referenced by any tensor dim in ``partition`` are
    treated as replica axes: the tensor is fully replicated along them.  The
    product of unused-axis sizes becomes ``num_replicas``, and each shard's
    ``replica_id`` is the row-major linearisation of its position over the
    unused-axis components of the mesh index.

    Args:
        name: Tensor name.
        dims: List of TensorDim describing the global tensor shape.
        mesh: Mesh grid of participants.
        partition: PartitionSpec mapping each dim to a mesh axis (or None).
        dtype: Torch dtype for the shard.

    Returns:
        List of TensorShardSpec, one per participant in the mesh (row-major).
    """
    if len(partition.specs) != len(dims):
        raise ValueError(
            f"PartitionSpec has {len(partition.specs)} entries but tensor "
            f"has {len(dims)} dims"
        )

    # Validate no duplicate axis names in the partition
    used_axes = [s for s in partition.specs if s is not None]
    if len(used_axes) != len(set(used_axes)):
        raise ValueError(f"Duplicate axis names in PartitionSpec: {partition.specs}")

    # Validate all axis names exist in mesh
    used_axes_set = set(used_axes)
    for axis in used_axes_set:
        if axis not in mesh.axis_names:
            raise ValueError(
                f"Axis {axis!r} in PartitionSpec not found in mesh "
                f"axes {mesh.axis_names}"
            )

    # Mesh axes not referenced by any dim become replica axes.
    unused_axis_indices = [
        i for i, name_ in enumerate(mesh.axis_names) if name_ not in used_axes_set
    ]
    unused_axis_sizes = [mesh.shape[i] for i in unused_axis_indices]
    num_replicas = 1
    for sz in unused_axis_sizes:
        num_replicas *= sz

    shards: List[TensorShardSpec] = []
    devices_array = mesh._devices

    for idx in np.ndindex(mesh.shape):
        participant = devices_array[idx]
        dim_specs = []
        for dim_i, dim in enumerate(dims):
            axis_name = partition.specs[dim_i]
            if axis_name is None:
                # Replicated: full range
                dim_specs.append(TensorDimSpec(dim.name, dim.size, 0, dim.size))
            else:
                axis_idx = mesh.axis_names.index(axis_name)
                axis_size = mesh.shape[axis_idx]
                pos = idx[axis_idx]
                chunk = dim.size // axis_size
                start = pos * chunk
                end = start + chunk
                dim_specs.append(TensorDimSpec(dim.name, dim.size, start, end))

        replica_id = 0
        for ai, sz in zip(unused_axis_indices, unused_axis_sizes):
            replica_id = replica_id * sz + idx[ai]

        shards.append(
            TensorShardSpec(
                name=name,
                dims=dim_specs,
                dtype=dtype,
                device=participant.device,
                replica_id=replica_id,
                num_replicas=num_replicas,
            )
        )

    return shards
