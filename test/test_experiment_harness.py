"""Tests for the experiment harness (Mesh, PartitionSpec, helpers, runner)."""

import uuid

import pytest
import torch

from setu._commons.datatypes import Device
from setu._coordinator import Participant, RegisterSet
from setu.cluster import ClusterSpec, DeviceSpec
from setu.experiment.mesh import Mesh, PartitionSpec, P


# ---------------------------------------------------------------------------
# Helpers for building test data
# ---------------------------------------------------------------------------

def _make_participant(node_idx: int, dev_idx: int) -> Participant:
    """Create a Participant with a deterministic UUID for *node_idx*."""
    node_id = uuid.UUID(int=node_idx)
    device = Device(torch_device=torch.device("cpu"))
    return Participant(node_id, device)


def _make_cluster_spec(
    num_nodes: int, devices_per_node: int, base_port: int = 30000
) -> ClusterSpec:
    """Create a minimal ClusterSpec for testing."""
    nodes = {}
    for n in range(num_nodes):
        node_id = uuid.UUID(int=n)
        device_specs = [
            DeviceSpec(Device(torch_device=torch.device("cpu")))
            for _ in range(devices_per_node)
        ]
        nodes[node_id] = (base_port + 100 + n, device_specs)
    return ClusterSpec(coordinator_port=base_port, nodes=nodes)


# ===========================================================================
# Mesh tests
# ===========================================================================


class TestMesh:
    def test_1d_mesh(self):
        p0 = _make_participant(0, 0)
        p1 = _make_participant(0, 1)
        mesh = Mesh([p0, p1], axis_names=("devices",))
        assert mesh.shape == (2,)
        assert mesh.ndim == 1
        assert mesh.axis_names == ("devices",)

    def test_2d_mesh(self):
        participants = [
            [_make_participant(n, d) for d in range(4)] for n in range(2)
        ]
        mesh = Mesh(participants, axis_names=("nodes", "devices"))
        assert mesh.shape == (2, 4)
        assert mesh.ndim == 2
        assert mesh.axis_names == ("nodes", "devices")

    def test_3d_mesh(self):
        participants = [
            [[_make_participant(r * 2 + n, d) for d in range(2)]
             for n in range(2)]
            for r in range(3)
        ]
        mesh = Mesh(participants, axis_names=("racks", "nodes", "devices"))
        assert mesh.shape == (3, 2, 2)
        assert mesh.ndim == 3
        assert mesh.axis_names == ("racks", "nodes", "devices")

    def test_from_cluster(self):
        spec = _make_cluster_spec(num_nodes=2, devices_per_node=4)
        mesh = Mesh.from_cluster(spec)
        assert mesh.shape == (2, 4)
        assert mesh.ndim == 2
        assert mesh.axis_names == ("nodes", "devices")

    def test_from_cluster_custom_axis_names(self):
        spec = _make_cluster_spec(num_nodes=3, devices_per_node=2)
        mesh = Mesh.from_cluster(spec, axis_names=("rack", "gpu"))
        assert mesh.axis_names == ("rack", "gpu")
        assert mesh.shape == (3, 2)

    def test_axis_size(self):
        spec = _make_cluster_spec(num_nodes=2, devices_per_node=4)
        mesh = Mesh.from_cluster(spec)
        assert mesh.axis_size("nodes") == 2
        assert mesh.axis_size("devices") == 4

    def test_axis_size_unknown_raises(self):
        mesh = Mesh([_make_participant(0, 0)], axis_names=("x",))
        with pytest.raises(ValueError, match="Unknown axis"):
            mesh.axis_size("bogus")

    def test_mismatched_ndim_raises(self):
        with pytest.raises(ValueError, match="dimensions"):
            Mesh(
                [[_make_participant(0, 0)]],
                axis_names=("a", "b", "c"),
            )


# ===========================================================================
# PartitionSpec tests
# ===========================================================================


class TestPartitionSpec:
    def test_construction(self):
        spec = PartitionSpec("x", "y", None)
        assert spec.specs == ("x", "y", None)

    def test_alias_P(self):
        spec = P("nodes", None)
        assert isinstance(spec, PartitionSpec)
        assert spec.specs == ("nodes", None)

    def test_empty(self):
        spec = P()
        assert spec.specs == ()

    def test_repr(self):
        spec = P("a", None, "b")
        assert "PartitionSpec" in repr(spec)
