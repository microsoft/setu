"""Helper functions for building shards and copy specs."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import numpy as np
import torch

from setu._commons.datatypes import (
    CopySpec,
    TensorDim,
    TensorDimSpec,
    TensorSelection,
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


# ---------------------------------------------------------------------------
# CopySpec builder
# ---------------------------------------------------------------------------


def build_copy_spec(
    src_name: str,
    dst_name: str,
    dims: List[TensorDim],
    src_selections: Optional[Dict[str, Union[Set[int], list]]] = None,
    dst_selections: Optional[Dict[str, Union[Set[int], list]]] = None,
) -> CopySpec:
    """Build a CopySpec from tensor dim descriptions and optional selections.

    Args:
        src_name: Source tensor name.
        dst_name: Destination tensor name.
        dims: List of TensorDim describing the tensor shape.
        src_selections: Optional dict mapping dim name -> index set to
            apply via ``.where()`` on the source selection.
        dst_selections: Optional dict mapping dim name -> index set to
            apply via ``.where()`` on the destination selection.

    Returns:
        A CopySpec ready for ``client.submit_pull()``.
    """
    dim_map = {d.name: d for d in dims}

    src_sel = TensorSelection(src_name, dim_map)
    if src_selections:
        for dim_name, indices in src_selections.items():
            src_sel = src_sel.where(dim_name, set(indices))

    dst_sel = TensorSelection(dst_name, dim_map)
    if dst_selections:
        for dim_name, indices in dst_selections.items():
            dst_sel = dst_sel.where(dim_name, set(indices))

    return CopySpec(src_name, dst_name, src_sel, dst_sel)
