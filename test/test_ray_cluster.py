"""
Integration tests for Setu Ray cluster management.

Tests the SetuCluster lifecycle, ClusterInfo structure, client connection,
and tensor registration through Ray-managed cluster components.
"""

import time

import pytest
import ray
import torch

from setu.ray import ClusterInfo, NodeAgentInfo, SetuCluster


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for the test module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    yield

    # Don't shut down Ray here — other tests may share the session


@pytest.fixture(scope="module")
def cluster(ray_context):
    """Start a SetuCluster for the test module and stop it after."""
    setu_cluster = SetuCluster()
    info = setu_cluster.start()

    yield setu_cluster, info

    setu_cluster.stop()


class TestSetuCluster:
    """Tests for SetuCluster lifecycle and cluster info."""

    @pytest.mark.gpu
    def test_cluster_starts_and_returns_info(self, cluster):
        """Test that the cluster starts and returns valid ClusterInfo."""
        _, info = cluster

        assert isinstance(info, ClusterInfo)
        assert info.coordinator_endpoint.startswith("tcp://")
        assert info.num_nodes >= 1
        assert info.total_gpus >= 1
        assert len(info.node_agents) >= 1

    @pytest.mark.gpu
    def test_node_agent_info_structure(self, cluster):
        """Test that each NodeAgentInfo has the expected fields."""
        _, info = cluster

        for na in info.node_agents:
            assert isinstance(na, NodeAgentInfo)
            assert na.node_id
            assert na.ip_address
            assert na.node_agent_endpoint.startswith("tcp://")
            assert na.num_gpus >= 1

    @pytest.mark.gpu
    def test_node_agent_endpoints_property(self, cluster):
        """Test the node_agent_endpoints convenience property."""
        _, info = cluster

        endpoints = info.node_agent_endpoints
        assert len(endpoints) == info.num_nodes
        for ep in endpoints:
            assert ep.startswith("tcp://")

    @pytest.mark.gpu
    def test_cluster_info_stored_on_instance(self, cluster):
        """Test that cluster_info property is available after start."""
        setu_cluster, info = cluster

        assert setu_cluster.cluster_info is info

    @pytest.mark.gpu
    def test_double_start_raises(self, cluster):
        """Test that starting an already-started cluster raises RuntimeError."""
        setu_cluster, _ = cluster

        with pytest.raises(RuntimeError, match="already started"):
            setu_cluster.start()

    @pytest.mark.gpu
    def test_actors_alive(self, cluster):
        """Test that all actors report alive after start."""
        setu_cluster, _ = cluster

        assert ray.get(setu_cluster._coordinator_actor.is_alive.remote())
        for actor in setu_cluster._node_agent_actors:
            assert ray.get(actor.is_alive.remote())


class TestClientConnection:
    """Tests for client connection through Ray-managed cluster."""

    @pytest.mark.gpu
    def test_client_connects_to_node_agent(self, cluster):
        """Test that a Client can connect to a Ray-managed NodeAgent."""
        from setu._client import Client

        _, info = cluster

        client = Client()
        client.connect(info.node_agent_endpoints[0])
        client.disconnect()

    @pytest.mark.gpu
    def test_register_tensor_through_ray_cluster(self, cluster):
        """Test registering a tensor shard through the Ray-managed cluster."""
        from setu._client import Client
        from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

        _, info = cluster

        client = Client()
        client.connect(info.node_agent_endpoints[0])

        device = Device(torch_device=torch.device("cuda:0"))
        dims = [
            TensorDimSpec("dim_0", 32, 0, 32),
            TensorDimSpec("dim_1", 64, 0, 64),
        ]
        shard_spec = TensorShardSpec(
            name="ray_test_tensor",
            dims=dims,
            dtype=torch.float32,
            device=device,
        )

        shard_ref = client.register_tensor_shard(shard_spec)
        assert shard_ref is not None
        assert shard_ref.name == "ray_test_tensor"

        client.disconnect()

    @pytest.mark.gpu
    def test_get_tensor_handle_through_ray_cluster(self, cluster):
        """Test getting a tensor handle through the Ray-managed cluster."""
        from setu._client import Client
        from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

        _, info = cluster

        client = Client()
        client.connect(info.node_agent_endpoints[0])

        device = Device(torch_device=torch.device("cuda:0"))
        dims = [
            TensorDimSpec("dim_0", 4, 0, 4),
            TensorDimSpec("dim_1", 8, 0, 8),
        ]
        shard_spec = TensorShardSpec(
            name="ray_handle_test_tensor",
            dims=dims,
            dtype=torch.float32,
            device=device,
        )

        shard_ref = client.register_tensor_shard(shard_spec)
        assert shard_ref is not None

        # Wait for allocation to complete
        time.sleep(0.5)

        handle = client.get_tensor_handle(shard_ref)
        assert handle is not None

        spec_dict = handle.to_dict()
        assert spec_dict["tensor_size"] == [4, 8]

        client.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--gpu"])
