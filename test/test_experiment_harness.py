"""Tests for the experiment harness (Mesh, PartitionSpec, helpers, runner)."""

import uuid
from test.fixtures.copy_spec_builder import build_copy_spec

import pytest
import torch

from setu._commons.datatypes import Device, TensorDim
from setu._coordinator import Link, Participant
from setu.cluster import ClusterSpec, DeviceSpec
from setu.cluster.mesh import Mesh, P, PartitionSpec
from setu.experiment.helpers import (
    ShardedTensor,
    shard_tensor,
)
from setu.experiment.runner import CopyMode, run_experiment

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


def _build_full_mesh(spec: ClusterSpec, intra: Link, inter: Link) -> "Topology":
    """Build a full-mesh Topology for testing."""
    from setu._coordinator import Topology

    topo = Topology()
    all_participants = []
    node_ids = []

    for node_id in sorted(spec.nodes.keys()):
        _, device_specs = spec.nodes[node_id]
        for ds in device_specs:
            all_participants.append(Participant(node_id, ds.device))
            node_ids.append(node_id)

    for i in range(len(all_participants)):
        for j in range(i + 1, len(all_participants)):
            link = intra if node_ids[i] == node_ids[j] else inter
            topo.add_bidirectional_link(all_participants[i], all_participants[j], link)

    return topo


def _build_one_level_spine(
    spec: ClusterSpec, intra: Link, inter_sym: Link, inter_cross: Link
) -> "Topology":
    """Build a one-level spine Topology for testing.

    Links between devices on the same node use *intra*.  Links between
    devices at the same position across nodes (symmetric) use *inter_sym*.
    All other cross-node links use *inter_cross*.
    """
    from setu._coordinator import Topology

    topo = Topology()
    entries = []
    for node_id in sorted(spec.nodes.keys()):
        _, device_specs = spec.nodes[node_id]
        for dev_idx, ds in enumerate(device_specs):
            entries.append((Participant(node_id, ds.device), node_id, dev_idx))

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            p_i, nid_i, didx_i = entries[i]
            p_j, nid_j, didx_j = entries[j]

            if nid_i == nid_j:
                link = intra
            elif didx_i == didx_j:
                link = inter_sym
            else:
                link = inter_cross

            topo.add_bidirectional_link(p_i, p_j, link)

    return topo


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


# ===========================================================================
# ShardedTensor tests
# ===========================================================================


class TestShardedTensor:
    def test_shards_property(self):
        """ShardedTensor.shards should produce the same result as shard_tensor()."""
        mesh = _make_mesh_2x4()
        dims = [TensorDim("page", 64), TensorDim("head", 8)]
        partition = P("x", "y")

        st = ShardedTensor(
            name="t", dims=dims, mesh=mesh, partition=partition, dtype=torch.float32
        )
        expected = shard_tensor("t", dims, mesh, partition, torch.float32)

        assert len(st.shards) == len(expected)
        for a, b in zip(st.shards, expected):
            assert a.name == b.name
            assert len(a.dims) == len(b.dims)
            for da, db in zip(a.dims, b.dims):
                assert da.name == db.name
                assert da.size == db.size
                assert da.start == db.start
                assert da.end == db.end

    def test_frozen(self):
        """ShardedTensor should be immutable (frozen dataclass)."""
        mesh = _make_mesh_2x4()
        dims = [TensorDim("a", 8)]
        st = ShardedTensor(name="t", dims=dims, mesh=mesh, partition=P("x"))
        with pytest.raises(AttributeError):
            st.name = "other"

    def test_default_dtype(self):
        """Default dtype should be torch.float32."""
        mesh = _make_mesh_2x4()
        dims = [TensorDim("a", 8)]
        st = ShardedTensor(name="t", dims=dims, mesh=mesh, partition=P("x"))
        assert st.dtype == torch.float32


# ===========================================================================
# Topology builder tests
# ===========================================================================


def _make_gpu_cluster_spec(
    num_nodes: int, devices_per_node: int, base_port: int = 30000
) -> ClusterSpec:
    """Create a ClusterSpec with unique CUDA-like devices for topology tests."""
    nodes = {}
    dev_counter = 0
    for n in range(num_nodes):
        node_id = uuid.UUID(int=n)
        device_specs = []
        for _ in range(devices_per_node):
            device_specs.append(
                DeviceSpec(Device(torch_device=torch.device(f"cuda:{dev_counter}")))
            )
            dev_counter += 1
        nodes[node_id] = (base_port + 100 + n, device_specs)
    return ClusterSpec(coordinator_port=base_port, nodes=nodes)


class TestBuildFullMesh:
    def test_edge_count(self):
        """Full mesh of N participants -> N*(N-1) directed edges."""
        spec = _make_gpu_cluster_spec(num_nodes=2, devices_per_node=2)
        intra = Link(0.0, 200.0)
        inter = Link(10.0, 100.0)
        topo = _build_full_mesh(spec, intra, inter)

        edges = topo.get_edges()
        n = 4  # 2 nodes * 2 devices
        assert len(edges) == n * (n - 1)

    def test_link_properties(self):
        """Intra links should differ from inter links."""
        spec = _make_gpu_cluster_spec(num_nodes=2, devices_per_node=2)
        intra = Link(0.0, 200.0)
        inter = Link(10.0, 100.0)
        topo = _build_full_mesh(spec, intra, inter)

        edges = topo.get_edges()

        for src, dst, link in edges:
            if src.node_id == dst.node_id:
                assert link.latency_us == 0.0
                assert link.bandwidth_gbps == 200.0
            else:
                assert link.latency_us == 10.0
                assert link.bandwidth_gbps == 100.0


class TestBuildOneLevelSpine:
    def test_symmetric_vs_cross(self):
        """Symmetric inter-node links (same dev idx) should differ from cross."""
        spec = _make_gpu_cluster_spec(num_nodes=2, devices_per_node=4)
        intra = Link(0.0, 200.0)
        inter_sym = Link(10.0, 100.0)
        inter_cross = Link(10.0, 50.0)
        topo = _build_one_level_spine(spec, intra, inter_sym, inter_cross)

        edges = topo.get_edges()
        n = 8  # 2 nodes * 4 devices
        assert len(edges) == n * (n - 1)

        sym_count = 0
        cross_count = 0
        for src, dst, link in edges:
            if src.node_id != dst.node_id:
                if link.bandwidth_gbps == 100.0:
                    sym_count += 1
                elif link.bandwidth_gbps == 50.0:
                    cross_count += 1

        # 4 symmetric pairs * 2 directions = 8
        assert sym_count == 8
        # 4*3 cross pairs * 2 directions = 24
        assert cross_count == 24


# ===========================================================================
# CopySpec builder tests
# ===========================================================================


class TestBuildCopySpec:
    def test_full_copy(self):
        """build_copy_spec without selections produces a full-tensor copy."""
        dims = [TensorDim("page", 64), TensorDim("head", 8)]
        cs = build_copy_spec("src", "dst", dims)
        assert cs.src_name == "src"
        assert cs.dst_name == "dst"
        assert cs.src_selection.is_spanning()
        assert cs.dst_selection.is_spanning()

    def test_partial_selections(self):
        """build_copy_spec with selections applies .where() correctly."""
        dims = [TensorDim("page", 64), TensorDim("head", 8)]
        cs = build_copy_spec(
            "src",
            "dst",
            dims,
            src_selections={"page": {0, 1, 2}},
            dst_selections={"page": {10, 11, 12}},
        )
        assert cs.src_name == "src"
        assert cs.dst_name == "dst"
        # With selections applied, should not be spanning
        assert not cs.src_selection.is_spanning()
        assert not cs.dst_selection.is_spanning()


# ===========================================================================
# run_experiment integration test (requires GPUs + Ray)
# ===========================================================================


@pytest.mark.gpu
def test_run_experiment_simple_1d_copy():
    """Simple 1D copy via run_experiment on a Ray cluster with 2+ GPUs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Need 2 CUDA devices, got {torch.cuda.device_count()}")

    import ray

    from setu.cluster.ray import Cluster

    if not ray.is_initialized():
        ray.init()

    cluster = Cluster()
    try:
        cluster_info = cluster.start()

        mesh = Mesh.from_cluster_info(cluster_info)
        dims = [TensorDim("dim0", 1024)]

        src = ShardedTensor(
            name="src_t",
            dims=dims,
            mesh=mesh,
            partition=P(None),
            dtype=torch.float32,
        )
        dst = ShardedTensor(
            name="dst_t",
            dims=dims,
            mesh=mesh,
            partition=P(None),
            dtype=torch.float32,
        )

        result = run_experiment(
            cluster_info=cluster_info,
            src=src,
            dst=dst,
            copy_mode=CopyMode.PULL,
            init_value=7.0,
        )

        assert result.success, f"Experiment failed: {result.errors}"
        assert result.elapsed_s > 0
        assert len(result.source_results) == len(src.shards)
        assert len(result.dest_results) == len(dst.shards)

    finally:
        cluster.stop()


# ===========================================================================
# run_experiment integration tests (MultiprocessingCluster)
# ===========================================================================


@pytest.mark.gpu
class TestExperimentRunnerSingleNode:
    """Integration tests for run_experiment using MultiprocessingCluster.

    Source mesh uses GPUs 0-1 (2 shards), destination mesh uses GPUs 0-3
    (4 shards).  The tensor is sharded along the single dimension so that
    the copy redistributes data across a different number of devices.
    """

    MIN_GPUS = 4
    # TENSOR_SIZE = 1024
    TENSOR_SIZE = 64 * 1024 * 1024

    @pytest.fixture(autouse=True)
    def _require_gpus(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < self.MIN_GPUS:
            pytest.skip(
                f"Need >= {self.MIN_GPUS} CUDA devices, got {torch.cuda.device_count()}"
            )

    def _make_cluster_spec(self, base_port: int = 50000) -> ClusterSpec:
        """Create a ClusterSpec for a single node with MIN_GPUS devices."""
        node_id = uuid.UUID(int=0)
        device_specs = [
            DeviceSpec(Device(torch_device=torch.device(f"cuda:{i}")))
            for i in range(self.MIN_GPUS)
        ]
        nodes = {node_id: (base_port + 100, device_specs)}
        return ClusterSpec(coordinator_port=base_port, nodes=nodes)

    def _build_src_dst(self, cluster):
        """Build sharded src (GPUs 0-1) and dst (GPUs 0-3) tensors."""
        node = cluster.cluster_info.nodes[0]
        node_id = uuid.UUID(node.node_id)
        participants = [Participant(node_id, dev) for dev in node.devices]

        src_mesh = Mesh([participants[0], participants[1]], axis_names=("devices",))
        dst_mesh = Mesh(
            [participants[0], participants[1], participants[2], participants[3]],
            axis_names=("devices",),
        )

        dims = [TensorDim("dim0", self.TENSOR_SIZE)]
        src = ShardedTensor(
            name="src_t",
            dims=dims,
            mesh=src_mesh,
            partition=P("devices"),
            dtype=torch.float32,
        )
        dst = ShardedTensor(
            name="dst_t",
            dims=dims,
            mesh=dst_mesh,
            partition=P("devices"),
            dtype=torch.float32,
        )
        return src, dst

    def _assert_success(self, result, src, dst):
        """Verify experiment completed successfully with matching values."""
        assert result.success, f"Experiment failed: {result.errors}"
        assert result.elapsed_s > 0
        assert len(result.source_results) == len(src.shards)
        assert len(result.dest_results) == len(dst.shards)
        for dr in result.dest_results:
            assert dr["values_match"], (
                f"Shard {dr['shard_name']}: expected={dr['expected_value']} "
                f"actual={dr['actual_value']}"
            )

    # -- PULL mode -----------------------------------------------------------

    def test_pull_no_selections(self):
        """PULL mode, no selections, src sharded over 2 GPUs -> dst over 4."""
        from setu.cluster.multiprocessing import Cluster as MultiprocessingCluster

        with MultiprocessingCluster(self._make_cluster_spec()) as cluster:
            src, dst = self._build_src_dst(cluster)

            result = run_experiment(
                cluster_info=cluster.cluster_info,
                src=src,
                dst=dst,
                copy_mode=CopyMode.PULL,
                init_value=7.0,
                timeout=120.0,
                n_copy_rounds=10,
                n_warmup_rounds=1,

            )
            print(result.pretty_print())
            self._assert_success(result, src, dst)

    def test_pull_with_selections(self):
        """PULL mode with slice selection covering the full dimension."""
        from setu.cluster.multiprocessing import Cluster as MultiprocessingCluster

        with MultiprocessingCluster(self._make_cluster_spec()) as cluster:
            src, dst = self._build_src_dst(cluster)

            result = run_experiment(
                cluster_info=cluster.cluster_info,
                src=src,
                dst=dst,
                copy_mode=CopyMode.PULL,
                init_value=3.0,
                selections={"dim0": slice(0, self.TENSOR_SIZE)},

            )
            self._assert_success(result, src, dst)

    # -- COPY mode -----------------------------------------------------------

    def test_copy_no_selections(self):
        """COPY (two-sided) mode, no selections, 2 GPUs -> 4 GPUs."""
        from setu.cluster.multiprocessing import Cluster as MultiprocessingCluster

        with MultiprocessingCluster(self._make_cluster_spec()) as cluster:
            src, dst = self._build_src_dst(cluster)

            result = run_experiment(
                cluster_info=cluster.cluster_info,
                src=src,
                dst=dst,
                copy_mode=CopyMode.COPY,
                init_value=7.0,

            )
            self._assert_success(result, src, dst)

    def test_copy_with_selections(self):
        """COPY (two-sided) mode with slice selection covering the full dim."""
        from setu.cluster.multiprocessing import Cluster as MultiprocessingCluster

        with MultiprocessingCluster(self._make_cluster_spec()) as cluster:
            src, dst = self._build_src_dst(cluster)

            result = run_experiment(
                cluster_info=cluster.cluster_info,
                src=src,
                dst=dst,
                copy_mode=CopyMode.COPY,
                init_value=3.0,
                selections={"dim0": slice(0, self.TENSOR_SIZE)},

            )
            self._assert_success(result, src, dst)
