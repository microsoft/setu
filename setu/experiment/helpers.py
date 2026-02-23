"""Helper functions for building shards, topologies, and copy specs."""

from typing import List

import numpy as np
import torch

from setu._commons.datatypes import (
    TensorDim,
    TensorDimSpec,
    TensorShardSpec,
)
from setu.cluster.mesh import Mesh, PartitionSpec


def shard_tensor(
    name: str,
    dims: List[TensorDim],
    mesh: Mesh,
    partition: PartitionSpec,
    dtype: torch.dtype = torch.float32,
) -> List[TensorShardSpec]:
    """Produce one TensorShardSpec per mesh position from a Mesh + PartitionSpec.

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
    for axis in used_axes:
        if axis not in mesh.axis_names:
            raise ValueError(
                f"Axis {axis!r} in PartitionSpec not found in mesh "
                f"axes {mesh.axis_names}"
            )

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

        shards.append(
            TensorShardSpec(
                name=name, dims=dim_specs, dtype=dtype, device=participant.device
            )
        )

    return shards
