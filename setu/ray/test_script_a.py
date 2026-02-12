"""
Client A: Start the Setu cluster, register a source shard on cuda:0,
fill it with 42.0, then wait for client B to pull.

Run on node A:
    python -m setu.ray.test_script_a
"""

import json
import signal
import sys
import time

import ray
import torch

from setu._client import Client
from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
from setu.ray import SetuCluster

CLUSTER_INFO_FILE = "/tmp/setu_cluster_info.json"


def _find_local_node_agent(info):
    """Find the node agent running on this node by matching IP."""
    local_ip = ray.util.get_node_ip_address()
    for na in info.node_agents:
        if na.ip_address == local_ip:
            return na
    raise RuntimeError(
        f"No node agent found for local IP {local_ip}. "
        f"Available: {[na.ip_address for na in info.node_agents]}"
    )


@ray.remote(num_gpus=1)
def fill_source_tensor(endpoint):
    """Register a source shard and fill it with 42.0.

    Runs as a Ray task with GPU access so CUDA IPC handles work.
    Blocks forever to keep the shard alive.
    """
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    client = Client()
    client.connect(endpoint)

    src_device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 4, 0, 4),
        TensorDimSpec("dim_1", 8, 0, 8),
    ]
    src_spec = TensorShardSpec(
        name="pull_src_tensor",
        dims=dims,
        dtype=torch.float32,
        device=src_device,
    )
    src_ref = client.register_tensor_shard(src_spec)
    assert src_ref is not None, "Failed to register source shard"

    client.wait_for_shard_allocation(src_ref.shard_id)
    print("Source shard allocated")

    ipc_spec, _, _ = client.get_tensor_handle(src_ref)
    tensor = rebuild_cuda_tensor(
        **ipc_spec.to_dict(),
        tensor_cls=torch.Tensor,
        storage_cls=torch.storage.UntypedStorage,
    )
    tensor.fill_(42.0)
    torch.cuda.synchronize()
    print(f"Source tensor filled with 42.0 (shape={list(tensor.shape)}, device={tensor.device})")

    # Block forever — shard must stay alive for the pull
    while True:
        time.sleep(1)


def main():
    # Connect to existing Ray cluster
    ray.init()

    # Start the Setu cluster
    setu_cluster = SetuCluster()
    info = setu_cluster.start()

    # Write cluster info so client B can connect
    cluster_data = {
        "coordinator_endpoint": info.coordinator_endpoint,
        "node_agents": [
            {
                "node_id": na.node_id,
                "ip_address": na.ip_address,
                "node_agent_endpoint": na.node_agent_endpoint,
                "num_gpus": na.num_gpus,
            }
            for na in info.node_agents
        ],
    }
    with open(CLUSTER_INFO_FILE, "w") as f:
        json.dump(cluster_data, f, indent=2)
    print(f"Cluster info written to {CLUSTER_INFO_FILE}")
    print(f"Coordinator: {info.coordinator_endpoint}")
    for na in info.node_agents:
        print(f"  NodeAgent: {na.node_agent_endpoint} ({na.num_gpus} GPUs, ip={na.ip_address})")

    # Find the node agent on this node
    local_na = _find_local_node_agent(info)
    print(f"Local node agent: {local_na.node_agent_endpoint} (ip={local_na.ip_address})")

    # Launch source shard registration as a Ray task with GPU access
    print(f"Launching source shard task on {local_na.node_agent_endpoint}")
    fill_ref = fill_source_tensor.remote(local_na.node_agent_endpoint)

    # Give the task time to register and fill
    time.sleep(5)

    print("\nClient A ready. Run test_script_b.py on the other node.")
    print("Press Ctrl+C to stop.\n")

    def handle_signal(_signum, _frame):
        print("\nShutting down...")
        ray.cancel(fill_ref, force=True)
        setu_cluster.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
