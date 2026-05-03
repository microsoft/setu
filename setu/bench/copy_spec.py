"""Loaders for benchmark tensor-spec and selections YAML files.

A *tensor-spec* file describes the src and dst tensors and their layouts
(mesh + partition + device placement) over a running cluster.

A *selections* file describes the per-dim narrowing applied to the src and
dst when issuing a copy.  The file always wraps copies in a list; each
entry produces one independent copy operation, executed sequentially in
file order.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

# Use the libyaml-backed C loader when available; the pure-Python loader
# is unusably slow on large selections.yaml files (10s of MB).
try:
    from yaml import CSafeLoader as _SafeLoader
except ImportError:
    from yaml import SafeLoader as _SafeLoader

from setu._commons.datatypes import TensorDim
from setu._coordinator import Participant
from setu.bench.cluster_setup import resolve_device_specs
from setu.bench.helpers import ShardedTensor
from setu.bench.runner import DimSelection
from setu.cluster.info import ClusterInfo
from setu.cluster.mesh import Mesh, P

# Mirrors the keys accepted by torch dtype names and what kv buffers commonly use.
_DTYPE_BY_NAME: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
}


def _parse_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key not in _DTYPE_BY_NAME:
        raise ValueError(
            f"Unknown dtype {name!r}; expected one of {sorted(_DTYPE_BY_NAME)}"
        )
    return _DTYPE_BY_NAME[key]


def _build_tensor(
    side: str,
    side_spec: Dict[str, Any],
    cluster_info: ClusterInfo,
) -> ShardedTensor:
    """Build a ShardedTensor from the ``src:`` / ``dst:`` block of tensor_spec.yaml."""
    for required in ("name", "dims", "layout", "dtype"):
        if required not in side_spec:
            raise ValueError(f"tensor_spec.{side}: missing required key {required!r}")

    name = side_spec["name"]
    dtype = _parse_dtype(side_spec["dtype"])

    dims_raw = side_spec["dims"]
    if not isinstance(dims_raw, list) or not dims_raw:
        raise ValueError(f"tensor_spec.{side}.dims must be a non-empty list")
    dims: List[TensorDim] = []
    for entry in dims_raw:
        if not isinstance(entry, dict) or "name" not in entry or "size" not in entry:
            raise ValueError(
                f"tensor_spec.{side}.dims entries must be mappings with "
                f"'name' and 'size' keys, got {entry!r}"
            )
        dims.append(TensorDim(entry["name"], int(entry["size"])))

    layout = side_spec["layout"]
    for required in ("devices", "mesh_shape", "axes", "partition"):
        if required not in layout:
            raise ValueError(
                f"tensor_spec.{side}.layout: missing required key {required!r}"
            )

    device_specs = layout["devices"]
    if not isinstance(device_specs, list) or not device_specs:
        raise ValueError(f"tensor_spec.{side}.layout.devices must be a non-empty list")
    participants: List[Participant] = resolve_device_specs(device_specs, cluster_info)

    mesh_shape = tuple(int(s) for s in layout["mesh_shape"])
    axes = tuple(str(a) for a in layout["axes"])
    if len(axes) != len(mesh_shape):
        raise ValueError(
            f"tensor_spec.{side}.layout: mesh_shape has {len(mesh_shape)} dims "
            f"but axes has {len(axes)} entries"
        )
    expected = 1
    for s in mesh_shape:
        expected *= s
    if expected != len(participants):
        raise ValueError(
            f"tensor_spec.{side}.layout: mesh_shape product {expected} does not "
            f"match number of resolved devices {len(participants)}"
        )
    if len(set(axes)) != len(axes):
        raise ValueError(f"tensor_spec.{side}.layout.axes contains duplicates: {axes}")

    import numpy as np

    devices_grid = np.array(participants, dtype=object).reshape(mesh_shape)
    mesh = Mesh(devices_grid, axis_names=axes)

    partition_raw = layout["partition"]
    if not isinstance(partition_raw, list) or len(partition_raw) != len(dims):
        raise ValueError(
            f"tensor_spec.{side}.layout.partition must have one entry per "
            f"tensor dim ({len(dims)}), got {partition_raw!r}"
        )
    partition = P(*[None if p is None else str(p) for p in partition_raw])

    return ShardedTensor(
        name=name, dims=dims, mesh=mesh, partition=partition, dtype=dtype
    )


def load_tensor_spec(
    path: str, cluster_info: ClusterInfo
) -> Tuple[ShardedTensor, ShardedTensor]:
    """Parse a tensor_spec.yaml into (src, dst) ShardedTensors.

    Asserts:
      * src and dst have identical dim names and sizes (in matching order).
      * src.dtype == dst.dtype.
    """
    raw = yaml.load(Path(path).read_text(), Loader=_SafeLoader)
    if not isinstance(raw, dict) or "src" not in raw or "dst" not in raw:
        raise ValueError(
            f"tensor_spec {path}: top-level must contain 'src' and 'dst' keys"
        )

    src = _build_tensor("src", raw["src"], cluster_info)
    dst = _build_tensor("dst", raw["dst"], cluster_info)

    if src.dtype != dst.dtype:
        raise ValueError(
            f"tensor_spec {path}: src.dtype ({src.dtype}) must equal dst.dtype "
            f"({dst.dtype}); dtype mismatch is unsupported"
        )

    src_shape = [(d.name, d.size) for d in src.dims]
    dst_shape = [(d.name, d.size) for d in dst.dims]
    if src_shape != dst_shape:
        raise ValueError(
            f"tensor_spec {path}: src and dst dims must match exactly; "
            f"src={src_shape}, dst={dst_shape}"
        )

    return src, dst


def _parse_dim_selection(dim_name: str, value: Any) -> DimSelection:
    """Parse one per-dim selection entry into the form runner expects.

    Mirrors the Python ``TensorSelection.where()`` API:
        int                  -> single index
        list[int]            -> explicit indices
        {start:a, end:b}     -> slice(a, b)
    """
    if isinstance(value, bool):
        # bool is an int subclass; reject explicitly to avoid surprises.
        raise ValueError(f"selection for dim {dim_name!r} cannot be a bool")
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in value):
            raise ValueError(
                f"selection list for dim {dim_name!r} must contain ints, "
                f"got {value!r}"
            )
        return list(value)
    if isinstance(value, dict):
        if set(value.keys()) != {"start", "end"}:
            raise ValueError(
                f"selection slice for dim {dim_name!r} must have exactly "
                f"keys {{'start','end'}}, got {sorted(value.keys())}"
            )
        start = int(value["start"])
        end = int(value["end"])
        if start >= end:
            raise ValueError(
                f"selection slice for dim {dim_name!r}: start ({start}) must be "
                f"less than end ({end})"
            )
        return slice(start, end)
    raise TypeError(
        f"selection for dim {dim_name!r}: expected int, list[int], or "
        f"{{start, end}} mapping; got {type(value).__name__}"
    )


@dataclass(frozen=True)
class CopySelections:
    """A single (src_selection, dst_selection) pair for one copy."""

    src: Dict[str, DimSelection]
    dst: Dict[str, DimSelection]


def load_selections(path: str) -> List[CopySelections]:
    """Parse a selections.yaml into a list of CopySelections, in file order.

    Every entry under ``copies:`` produces one CopySelections; callers run
    them sequentially as independent copy operations.
    """
    raw = yaml.load(Path(path).read_text(), Loader=_SafeLoader)
    if not isinstance(raw, dict) or "copies" not in raw:
        raise ValueError(f"selections {path}: top-level must contain a 'copies' list")

    copies = raw["copies"]
    if not isinstance(copies, list) or not copies:
        raise ValueError(f"selections {path}: 'copies' must be a non-empty list")

    def _parse_side(idx: int, side: str, raw_side: Any) -> Dict[str, DimSelection]:
        if not isinstance(raw_side, dict):
            raise ValueError(
                f"selections {path}: copies[{idx}].{side} must be a mapping of "
                f"dim_name -> selection"
            )
        return {
            str(dim_name): _parse_dim_selection(str(dim_name), value)
            for dim_name, value in raw_side.items()
        }

    parsed: List[CopySelections] = []
    for idx, entry in enumerate(copies):
        if not isinstance(entry, dict) or "src" not in entry or "dst" not in entry:
            raise ValueError(
                f"selections {path}: copies[{idx}] must have 'src' and 'dst' "
                f"mappings"
            )
        parsed.append(
            CopySelections(
                src=_parse_side(idx, "src", entry["src"]),
                dst=_parse_side(idx, "dst", entry["dst"]),
            )
        )
    return parsed
