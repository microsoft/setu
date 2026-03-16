"""
Test poll-based completion for copy operations.

Tests the shared-memory SPSC completion ring:
- Dest clients poll for completions instead of blocking on WaitForCopy
- Verifies that PollCompletions returns the correct CopyOperationIds
"""

import uuid
from test.fixtures import (
    ClusterSpec,
    DeviceSpec,
    SetuTestCluster,
    run_polling_dest_client,
    run_source_client,
)

import pytest
import torch

from setu._commons.datatypes import Device


def _make_dims_data(dim_names, dim_sizes, num_shards, i, shard_dim=0):
    dim_owned_range = []
    for idx, sz in enumerate(dim_sizes):
        if idx == shard_dim:
            shard_sz = sz // num_shards
            dim_owned_range.append((i * shard_sz, (i + 1) * shard_sz))
        else:
            dim_owned_range.append((0, sz))
    return [
        (n, sz, s, e) for (n, sz, (s, e)) in zip(dim_names, dim_sizes, dim_owned_range)
    ]


@pytest.mark.gpu
def test_poll_multiple_completions():
    """
    Test poll-based completion with multiple source and destination clients.

    Setup:
    - 1 NodeAgent (device 0)
    - 2 source clients on cuda:0 (register source shards, initialize to 10.0)
    - 4 destination clients on cuda:0 (register dest shards, submit pulls,
      poll for completions)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    node_id = uuid.uuid4()
    init_value = 10.0
    dim_names = ["a", "b"]
    dim_sizes = [4, 4]

    spec = ClusterSpec(
        coordinator_port=29502,
        nodes={node_id: (29602, [DeviceSpec(Device(torch.device("cuda:0")))])},
    )

    with SetuTestCluster(spec) as cluster:
        ctx = cluster.mp_context
        source_init_events = [ctx.Event() for _ in range(2)]
        all_sources_ready = ctx.Event()
        source_results = ctx.Queue()
        dest_results = ctx.Queue()

        processes = []

        try:
            # Start 2 source clients
            for i in range(2):
                proc = ctx.Process(
                    target=run_source_client,
                    args=(
                        cluster.client_endpoint(node_id),
                        "source_tensor",
                        _make_dims_data(dim_names, dim_sizes, 2, i),
                        init_value,
                        source_init_events[i],
                        source_results,
                        i,
                    ),
                )
                proc.start()
                processes.append(proc)

            for i, event in enumerate(source_init_events):
                assert event.wait(timeout=30), f"Source client {i} failed to initialize"
            all_sources_ready.set()

            # Start 4 destination clients using polling
            for i in range(4):
                proc = ctx.Process(
                    target=run_polling_dest_client,
                    args=(
                        cluster.client_endpoint(node_id),
                        "source_tensor",
                        "dest_tensor",
                        _make_dims_data(dim_names, dim_sizes, 4, i),
                        all_sources_ready,
                        init_value,
                        dest_results,
                        i,
                    ),
                )
                proc.start()
                processes.append(proc)

            # Collect source results
            for _ in range(2):
                result = source_results.get(timeout=30)
                assert result["success"], f"Source client failed: {result.get('error')}"

            # Collect and verify destination results
            for _ in range(4):
                result = dest_results.get(timeout=60)
                assert result["success"], (
                    f"Dest client {result.get('client_id')} failed: "
                    f"{result.get('error')}\n{result.get('traceback', '')}"
                )
                assert result["values_match"], (
                    f"Dest client {result['client_id']}: "
                    f"expected {result['expected_value']}, got {result['actual_value']}"
                )

        finally:
            for proc in processes:
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()


@pytest.mark.gpu
def test_poll_completions_empty_returns_empty():
    """
    Test that poll_completions returns an empty list when no copies are pending.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    node_id = uuid.uuid4()

    spec = ClusterSpec(
        coordinator_port=29503,
        nodes={node_id: (29603, [DeviceSpec(Device(torch.device("cuda:0")))])},
    )

    with SetuTestCluster(spec) as cluster:
        client = Client()
        client.connect(cluster.client_endpoint(node_id))

        # Register a shard so the client is fully operational
        dims_spec = [TensorDimSpec("x", 4, 0, 4)]
        device = Device(torch_device=torch.device("cuda:0"))
        shard_spec = TensorShardSpec(
            name="poll_test_tensor", dims=dims_spec, dtype=torch.float32, device=device
        )
        shard_ref = client.register_tensor_shard(shard_spec)
        assert shard_ref is not None
        client.wait_for_shard_allocation(shard_ref.shard_id)

        # Poll with no pending copies — should return empty list
        completed = client.poll_completions()
        assert isinstance(completed, list)
        assert len(completed) == 0

        client.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
