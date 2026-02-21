"""Mesh and PartitionSpec: JAX-style named grid of participants."""

from typing import Optional, Tuple

import numpy as np

from setu._coordinator import Participant
from setu.cluster import ClusterSpec


class Mesh:
    """Named N-dimensional grid of Participants. Backed by numpy object array.

    Example::

        mesh = Mesh(
            [[p_n0_d0, p_n0_d1], [p_n1_d0, p_n1_d1]],
            axis_names=("nodes", "devices"),
        )  # shape (2, 2)
    """

    def __init__(self, devices, axis_names: Tuple[str, ...]):
        self._devices = np.array(devices, dtype=object)
        self._axis_names = axis_names
        if self._devices.ndim != len(axis_names):
            raise ValueError(
                f"devices has {self._devices.ndim} dimensions but "
                f"{len(axis_names)} axis names were given"
            )

    @classmethod
    def from_cluster(
        cls,
        spec: ClusterSpec,
        axis_names: Tuple[str, ...] = ("nodes", "devices"),
    ) -> "Mesh":
        """Build a 2D Mesh from a ClusterSpec.

        Rows correspond to nodes (ordered by node_id), columns to devices
        within each node.
        """
        rows = []
        for node_id in sorted(spec.nodes.keys()):
            _, device_specs = spec.nodes[node_id]
            row = [Participant(node_id, ds.device) for ds in device_specs]
            rows.append(row)
        return cls(rows, axis_names)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._devices.shape)

    @property
    def ndim(self) -> int:
        return self._devices.ndim

    @property
    def axis_names(self) -> Tuple[str, ...]:
        return self._axis_names

    def axis_size(self, axis_name: str) -> int:
        """Return the size of a named axis."""
        idx = self._axis_index(axis_name)
        return self._devices.shape[idx]

    def _axis_index(self, axis_name: str) -> int:
        try:
            return self._axis_names.index(axis_name)
        except ValueError:
            raise ValueError(
                f"Unknown axis {axis_name!r}; " f"mesh has axes {self._axis_names}"
            )


class PartitionSpec:
    """Positional mapping of tensor dims to mesh axes.

    Each entry is either a mesh axis name (str) or ``None`` for replicated.

    Example::

        spec = PartitionSpec("nodes", "devices", None)
        # dim 0 → sharded along "nodes", dim 1 → "devices", dim 2 → replicated
    """

    def __init__(self, *specs: Optional[str]):
        self.specs = specs

    def __repr__(self) -> str:
        inner = ", ".join(repr(s) for s in self.specs)
        return f"PartitionSpec({inner})"


P = PartitionSpec
