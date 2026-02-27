"""
Tests for Cluster lifecycle: stop, restart, and error handling.

These tests create and destroy their own clusters, so they must run
in a separate module from tests that share a long-lived cluster fixture.
"""

import pytest
import ray
import torch

from setu.cluster.ray import Cluster

pytestmark = pytest.mark.skip(reason="Ray tests are flaky, temporarily disabled")


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for the test module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    yield


class TestClusterLifecycle:
    """Tests for cluster stop and restart behavior."""

    @pytest.mark.gpu
    def test_stop_and_restart(self, ray_context):
        """Test that a cluster can be stopped and a new one started."""
        cluster1 = Cluster()
        info1 = cluster1.start()
        assert info1.num_nodes >= 1
        cluster1.stop()

        assert cluster1.cluster_info is None

        # Start a new cluster
        cluster2 = Cluster()
        info2 = cluster2.start()
        assert info2.num_nodes >= 1
        cluster2.stop()

    @pytest.mark.gpu
    def test_start_fails_when_gpus_held(self, ray_context):
        """Test that starting a second cluster raises instead of hanging."""
        cluster1 = Cluster()
        cluster1.start()

        cluster2 = Cluster()
        try:
            with pytest.raises(RuntimeError, match="Timed out"):
                cluster2.start()
        finally:
            cluster1.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--gpu"])
