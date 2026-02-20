"""
Test multi-client pull operation with separate processes.

Tests the flow where:
- 2 source clients register source shards and initialize data
- 4 destination clients register destination shards and pull data
"""

import uuid
from test.fixtures import (
    ClusterSpec,
    DeviceSpec,
    SetuTestCluster,
    run_dest_client,
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
def test_multi_client_pull_same_device():
    """
    Test pull operation with multiple source and destination clients on the same device.

    Setup:
    - 1 NodeAgent (device 0)
    - 2 source clients on cuda:0 (register source shards, initialize to 10.0)
    - 4 destination clients on cuda:0 (register dest shards, pull data, verify value)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    node_id = uuid.uuid4()
    init_value = 10.0
    dim_names = ["a", "b"]
    dim_sizes = [4, 4]

    spec = ClusterSpec(
        coordinator_port=29500,
        nodes={node_id: (29600, [DeviceSpec(Device(torch.device("cuda:0")))])},
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

            # Start 4 destination clients
            for i in range(4):
                proc = ctx.Process(
                    target=run_dest_client,
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
def test_multi_client_pull_multi_device():
    """
    Test pull operation with multiple source and destination clients across 4 CUDA devices.

    Setup:
    - 1 NodeAgent with devices [0, 1, 2, 3]
    - 2 source clients: client 0 on cuda:0, client 1 on cuda:1
    - 4 destination clients: client i on cuda:i (i=0..3)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 4:
        pytest.skip(f"Need at least 4 CUDA devices, got {torch.cuda.device_count()}")

    node_id = uuid.uuid4()
    init_value = 10.0
    dim_names = ["a", "b"]
    dim_sizes = [4, 4]

    devs = [Device(torch.device(f"cuda:{i}")) for i in range(4)]
    spec = ClusterSpec(
        coordinator_port=29501,
        nodes={node_id: (29601, [DeviceSpec(d) for d in devs])},
    )

    with SetuTestCluster(spec) as cluster:
        ctx = cluster.mp_context
        source_init_events = [ctx.Event() for _ in range(2)]
        all_sources_ready = ctx.Event()
        source_results = ctx.Queue()
        dest_results = ctx.Queue()

        processes = []

        try:
            # Start 2 source clients on different devices
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
                        i,  # device_index: 0, 1
                    ),
                )
                proc.start()
                processes.append(proc)

            for i, event in enumerate(source_init_events):
                assert event.wait(timeout=30), f"Source client {i} failed to initialize"
            all_sources_ready.set()

            # Start 4 destination clients, each on a different device
            for i in range(4):
                proc = ctx.Process(
                    target=run_dest_client,
                    args=(
                        cluster.client_endpoint(node_id),
                        "source_tensor",
                        "dest_tensor",
                        _make_dims_data(dim_names, dim_sizes, 4, i),
                        all_sources_ready,
                        init_value,
                        dest_results,
                        i,
                        i,  # device_index: 0, 1, 2, 3
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
