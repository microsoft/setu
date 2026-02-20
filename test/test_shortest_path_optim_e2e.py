"""
E2E test for shortest-path-optimized copy across two nodes.

Setup:
- Node 0: 4 devices (cuda:0..3), hosts tensor_a sharded by 2 on devices 0,1
- Node 1: 4 devices (cuda:4..7), hosts tensor_b sharded by 4 on devices 4,5,6,7
- Topology: full mesh between all 8 participants
    - Intra-node: latency=0us, bw=200 Gbps
    - Inter-node symmetric (i <-> i+4): latency=10us, bw=100 Gbps
    - Inter-node cross (i <-> j+4, i!=j): latency=10us, bw=50 Gbps
- Planner: NCCL backend + ShortestPathRouting pass
- Pull-based copy: each tensor_b shard pulls from tensor_a
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
from setu._coordinator import Link, Participant, RegisterSet, Topology


def _make_dims_data(dim_name, dim_global_size, num_shards, shard_idx):
    shard_size = dim_global_size // num_shards
    start = shard_idx * shard_size
    end = start + shard_size
    return [(dim_name, dim_global_size, start, end)]


@pytest.mark.gpu
def test_shortest_path_optim_e2e():
    """
    E2E pull with shortest-path routing across 2 nodes / 8 devices.

    - tensor_a on node0, sharded [0..512) on cuda:0 and [512..1024) on cuda:1
    - tensor_b on node1, sharded across cuda:4, cuda:5, cuda:6, cuda:7
    - Each tensor_b shard pulls its slice from tensor_a
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 8:
        pytest.skip(f"Need 8 CUDA devices, got {torch.cuda.device_count()}")

    n0_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    n1_id = uuid.UUID("00000000-0000-0000-0000-000000000001")

    dim_name = "dim0"
    dim_global_size = 1024
    num_src_shards = 2
    num_dst_shards = 4
    init_value = 42.0

    # Build topology with real C++ types
    devs = [Device(torch.device(f"cuda:{i}")) for i in range(8)]

    topo = Topology()
    # Intra-node0
    for i in range(4):
        for j in range(i + 1, 4):
            topo.add_bidirectional_link(
                Participant(n0_id, devs[i]), Participant(n0_id, devs[j]), Link(0, 200)
            )
    # Intra-node1
    for i in range(4, 8):
        for j in range(i + 1, 8):
            topo.add_bidirectional_link(
                Participant(n1_id, devs[i]), Participant(n1_id, devs[j]), Link(0, 200)
            )
    # Inter-node: symmetric pairs (i <-> i+4) faster than cross pairs
    for i in range(4):
        for j in range(4, 8):
            symmetric = (j - 4) == i
            topo.add_bidirectional_link(
                Participant(n0_id, devs[i]),
                Participant(n1_id, devs[j]),
                Link(10, 100 if symmetric else 50),
            )

    # Build cluster spec
    reg = RegisterSet.uniform(1, 1024 * 1024)
    n0_device_specs = [DeviceSpec(devs[i], reg) for i in range(4)]
    n1_device_specs = [DeviceSpec(devs[i], reg) for i in range(4, 8)]

    spec = ClusterSpec(
        coordinator_port=29510,
        nodes={
            n0_id: (29610, n0_device_specs),
            n1_id: (29611, n1_device_specs),
        },
        topology=topo,
    )

    with SetuTestCluster(spec) as cluster:
        ctx = cluster.mp_context
        source_init_events = [ctx.Event() for _ in range(num_src_shards)]
        all_sources_ready = ctx.Event()
        source_results = ctx.Queue()
        dest_results = ctx.Queue()

        processes = []

        try:
            # Source clients: tensor_a on node0, devices 0 and 1
            for i in range(num_src_shards):
                proc = ctx.Process(
                    target=run_source_client,
                    args=(
                        cluster.client_endpoint(n0_id),
                        "tensor_a",
                        _make_dims_data(dim_name, dim_global_size, num_src_shards, i),
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
                assert event.wait(timeout=30), f"Source {i} failed to initialize"
            all_sources_ready.set()
            print("[Main] All sources ready", flush=True)

            for _ in range(num_src_shards):
                result = source_results.get(timeout=30)
                assert result["success"], f"Source failed: {result.get('error')}"

            # Dest clients: tensor_b on node1, devices 4,5,6,7
            for i in range(num_dst_shards):
                proc = ctx.Process(
                    target=run_dest_client,
                    args=(
                        cluster.client_endpoint(n1_id),
                        "tensor_a",
                        "tensor_b",
                        _make_dims_data(dim_name, dim_global_size, num_dst_shards, i),
                        all_sources_ready,
                        init_value,
                        dest_results,
                        i,
                        4 + i,  # device_index: 4, 5, 6, 7
                    ),
                )
                proc.start()
                processes.append(proc)

            for _ in range(num_dst_shards):
                result = dest_results.get(timeout=60)
                print(f"[Main] Dest result: {result}", flush=True)
                assert result["success"], (
                    f"Dest {result.get('client_id')} failed: "
                    f"{result.get('error')}\n{result.get('traceback', '')}"
                )
                assert result["values_match"], (
                    f"Dest {result['client_id']}: "
                    f"expected {result['expected_value']}, got {result['actual_value']}"
                )

            print("[Main] All dest clients verified!", flush=True)

        finally:
            for proc in processes:
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
