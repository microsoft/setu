"""
Python wrapper for TensorSelection with numpy-like indexing support.
"""

from __future__ import annotations

from typing import List, Union

from setu._commons.datatypes import TensorSelection as TensorSelection_C
from setu._commons.datatypes import TensorSlice
from setu.core.types import TensorName


class TensorSelection:
    """
    Python wrapper for TensorSelection with numpy-like fancy indexing.

    Supports:
    - Integer indexing: .where("dim", 5)
    - Slice indexing: .where("dim", slice(0, 10)) or equivalently [0:10]
    - List indexing: .where("dim", [1, 3, 5, 7])
    - Set indexing: .where("dim", {1, 3, 5, 7})
    """

    def __init__(self, native_selection: TensorSelection_C) -> None:
        """
        Initialize wrapper around native TensorSelection.

        Args:
            native_selection: Native C++ TensorSelection object
        """
        self._native = native_selection

    @property
    def name(self) -> TensorName:
        """Get tensor name."""
        return self._native.name

    @property
    def native(self) -> TensorSelection_C:
        """Get underlying native TensorSelection object."""
        return self._native

    def where(
        self,
        dim_name: TensorName,
        indices: Union[int, slice, List[int], set],
    ) -> TensorSelection:
        """
        Create new selection with specified indices for a dimension.

        Supports numpy-like fancy indexing:
        - Integer: Select single index
        - Slice: Select range of indices
        - List/Set: Select specific indices

        Args:
            dim_name: Name of the dimension to select
            indices: Index specification (int, slice, list, or set)

        Returns:
            New TensorSelection with the specified indices

        Examples:
            >>> # Single index
            >>> sel.where("page", 5)
            >>>
            >>> # Slice (like numpy [0:32])
            >>> sel.where("seq", slice(0, 32))
            >>>
            >>> # List of indices
            >>> sel.where("page", [1, 3, 5, 7])
            >>>
            >>> # Set of indices
            >>> sel.where("head", {0, 1, 2, 3})
        """
        if isinstance(indices, int):
            # Single integer: convert to set
            index_set = {indices}
            new_native = self._native.where(dim_name, index_set)
        elif isinstance(indices, slice):
            # Python slice: convert to TensorSlice
            start = indices.start if indices.start is not None else 0
            # Note: stop is required for TensorSlice
            if indices.stop is None:
                raise ValueError(
                    f"Slice for dimension '{dim_name}' must have explicit stop value"
                )
            stop = indices.stop
            if indices.step is not None and indices.step != 1:
                raise ValueError(
                    f"Slice step not supported (got step={indices.step}), use list of indices instead"
                )
            tensor_slice = TensorSlice(dim_name, start, stop)
            new_native = self._native.where(dim_name, tensor_slice)
        elif isinstance(indices, (list, tuple)):
            # List/tuple: convert to set
            index_set = set(indices)
            new_native = self._native.where(dim_name, index_set)
        elif isinstance(indices, set):
            # Already a set
            new_native = self._native.where(dim_name, indices)
        else:
            raise TypeError(
                f"Unsupported index type for dimension '{dim_name}': {type(indices)}. "
                f"Expected int, slice, list, or set"
            )

        return TensorSelection(new_native)

    def get_intersection(self, other: TensorSelection) -> TensorSelection:
        """
        Get intersection with another tensor selection.

        Args:
            other: Another TensorSelection to intersect with

        Returns:
            New TensorSelection representing the intersection

        Example:
            >>> sel1 = client.select("tensor").where("page", [0, 1, 2])
            >>> sel2 = client.select("tensor").where("page", [1, 2, 3])
            >>> intersection = sel1.get_intersection(sel2)  # page=[1, 2]
        """
        new_native = self._native.get_intersection(other._native)
        return TensorSelection(new_native)

    def is_spanning(self) -> bool:
        """Check if selection spans all dimensions."""
        return self._native.is_spanning()

    def is_empty(self) -> bool:
        """Check if selection is empty."""
        return self._native.is_empty()

    def is_compatible(self, other: TensorSelection) -> bool:
        """Check if compatible with another selection."""
        return self._native.is_compatible(other._native)

    def __str__(self) -> str:
        """String representation."""
        return str(self._native)

    def __repr__(self) -> str:
        """Repr representation."""
        return repr(self._native)
