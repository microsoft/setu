"""Test helper for building CopySpec objects from tensor dim descriptions."""

from typing import Dict, List, Optional, Set, Union

from setu._commons.datatypes import (
    CopySpec,
    TensorDim,
    TensorSelection,
)


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
