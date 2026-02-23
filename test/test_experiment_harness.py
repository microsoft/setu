"""Tests for the experiment harness (Mesh, PartitionSpec, helpers, runner)."""

import uuid

import pytest
import torch

from setu._commons.datatypes import Device
from setu._coordinator import Participant
from setu.cluster import ClusterSpec, DeviceSpec
from setu.cluster.mesh import Mesh, P, PartitionSpec
from setu.experiment.helpers import shard_tensor

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
        participants = [[_make_participant(n, d) for d in range(4)] for n in range(2)]
        mesh = Mesh(participants, axis_names=("nodes", "devices"))
        assert mesh.shape == (2, 4)
        assert mesh.ndim == 2
        assert mesh.axis_names == ("nodes", "devices")

    def test_3d_mesh(self):
        participants = [
            [[_make_participant(r * 2 + n, d) for d in range(2)] for n in range(2)]
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


# ===========================================================================
# shard_tensor tests
# ===========================================================================


def _make_mesh_2x4():
    """Build a (2, 4) mesh with distinct participants."""
    participants = [[_make_participant(n, d) for d in range(4)] for n in range(2)]
    return Mesh(participants, axis_names=("x", "y"))


class TestShardTensor:
    def test_full_sharding_2d(self):
        """P('x', 'y', None) on (2, 4) mesh -> 8 shards."""
        mesh = _make_mesh_2x4()
        dims = [TensorDim("page", 64), TensorDim("head", 8), TensorDim("hd", 128)]
        shards = shard_tensor("t", dims, mesh, P("x", "y", None))

        assert len(shards) == 8
        for s in shards:
            assert s.name == "t"
            assert s.get_num_dims() == 3
            # page: 64 / 2 = 32 each
            assert s.dims[0].size == 64
            assert s.dims[0].end - s.dims[0].start == 32
            # head: 8 / 4 = 2 each
            assert s.dims[1].size == 8
            assert s.dims[1].end - s.dims[1].start == 2
            # hd: replicated -> full range
            assert s.dims[2].start == 0
            assert s.dims[2].end == 128

    def test_swapped_axes(self):
        """P('y', 'x', None) -- axes swapped, different chunk sizes."""
        mesh = _make_mesh_2x4()
        dims = [TensorDim("page", 64), TensorDim("head", 8), TensorDim("hd", 128)]
        shards = shard_tensor("t", dims, mesh, P("y", "x", None))

        assert len(shards) == 8
        for s in shards:
            # page: 64 / 4 = 16 each (mapped to "y")
            assert s.dims[0].end - s.dims[0].start == 16
            # head: 8 / 2 = 4 each (mapped to "x")
            assert s.dims[1].end - s.dims[1].start == 4
            # hd: replicated
            assert s.dims[2].start == 0
            assert s.dims[2].end == 128

    def test_partial_sharding(self):
        """P('x', None, None) -- only first dim sharded along 'x'."""
        mesh = _make_mesh_2x4()
        dims = [TensorDim("page", 64), TensorDim("head", 8), TensorDim("hd", 128)]
        shards = shard_tensor("t", dims, mesh, P("x", None, None))

        assert len(shards) == 8
        for s in shards:
            # page sharded by x (size 2): 32 each
            assert s.dims[0].end - s.dims[0].start == 32
            # head and hd replicated
            assert s.dims[1].start == 0 and s.dims[1].end == 8
            assert s.dims[2].start == 0 and s.dims[2].end == 128

    def test_full_replication(self):
        """P(None, None, None) -- all shards identical (fully replicated)."""
        mesh = _make_mesh_2x4()
        dims = [TensorDim("page", 64), TensorDim("head", 8), TensorDim("hd", 128)]
        shards = shard_tensor("t", dims, mesh, P(None, None, None))

        assert len(shards) == 8
        for s in shards:
            assert s.dims[0].start == 0 and s.dims[0].end == 64
            assert s.dims[1].start == 0 and s.dims[1].end == 8
            assert s.dims[2].start == 0 and s.dims[2].end == 128

    def test_3d_mesh(self):
        """3D mesh with P('rack', 'node', 'device') -> 8 shards."""
        participants = [
            [[_make_participant(r * 2 + n, d) for d in range(2)] for n in range(2)]
            for r in range(2)
        ]
        mesh = Mesh(participants, axis_names=("rack", "node", "device"))
        dims = [TensorDim("a", 16), TensorDim("b", 8), TensorDim("c", 4)]
        shards = shard_tensor("t", dims, mesh, P("rack", "node", "device"))

        assert len(shards) == 8
        for s in shards:
            assert s.dims[0].end - s.dims[0].start == 8  # 16 / 2
            assert s.dims[1].end - s.dims[1].start == 4  # 8 / 2
            assert s.dims[2].end - s.dims[2].start == 2  # 4 / 2

    def test_shard_ranges_cover_full_dim(self):
        """All shard ranges along a sharded dim should tile the full extent."""
        mesh = _make_mesh_2x4()
        dims = [TensorDim("page", 64), TensorDim("head", 8), TensorDim("hd", 128)]
        shards = shard_tensor("t", dims, mesh, P("x", "y", None))

        # Collect unique (start, end) ranges for "head" dim (sharded along y=4)
        head_ranges = sorted({(s.dims[1].start, s.dims[1].end) for s in shards})
        assert head_ranges == [(0, 2), (2, 4), (4, 6), (6, 8)]

    def test_duplicate_axis_raises(self):
        mesh = _make_mesh_2x4()
        dims = [TensorDim("a", 8), TensorDim("b", 8)]
        with pytest.raises(ValueError, match="Duplicate"):
            shard_tensor("t", dims, mesh, P("x", "x"))

    def test_unknown_axis_raises(self):
        mesh = _make_mesh_2x4()
        dims = [TensorDim("a", 8)]
        with pytest.raises(ValueError, match="not found"):
            shard_tensor("t", dims, mesh, P("bogus"))

    def test_wrong_spec_length_raises(self):
        mesh = _make_mesh_2x4()
        dims = [TensorDim("a", 8), TensorDim("b", 4)]
        with pytest.raises(ValueError, match="entries"):
            shard_tensor("t", dims, mesh, P("x"))
