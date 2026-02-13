"""
Client B: Connect to an existing Setu cluster, register a destination
shard on cuda:0, then pull from the source shard registered by client A.

Run on node B (after test_script_a.py is ready):
    python -m setu.ray.test_script_b --cluster-info /tmp/setu_cluster_info.json

If the cluster info file is on a different node, copy it over or pass
the node agent endpoint directly:
    python -m setu.ray.test_script_b --endpoint tcp://<node_b_ip>:<port>
"""

import argparse
import json
import socket
import sys

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from setu._client import Client
from setu._commons.datatypes import (
    CopySpec,
    Device,
    TensorDim,
    TensorDimSpec,
    TensorSelection,
    TensorShardSpec,
)

CLUSTER_INFO_FILE = "/tmp/setu_cluster_info.json"


def _get_local_ip():
    """Get the local IP address (same method Ray uses)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def _find_local_endpoint(cluster_data):
    """Find the node agent on this node by matching IP."""
    local_ip = _get_local_ip()
    node_agents = cluster_data["node_agents"]

    print(f"Local IP: {local_ip}")
    print(f"Cluster has {len(node_agents)} node agent(s):")
    for i, na in enumerate(node_agents):
        marker = " <-- LOCAL" if na["ip_address"] == local_ip else ""
        print(f"  [{i}] {na['node_agent_endpoint']} ({na['num_gpus']} GPUs, ip={na['ip_address']}){marker}")

    for na in node_agents:
        if na["ip_address"] == local_ip:
            print(f"Using local node agent: {na['node_agent_endpoint']}")
            return na["node_agent_endpoint"]

    raise RuntimeError(
        f"No node agent found for local IP {local_ip}. "
        f"Available IPs: {[na['ip_address'] for na in node_agents]}"
    )


def load_endpoint(args):
    """Resolve the node agent endpoint to connect to."""
    if args.endpoint:
        return args.endpoint

    with open(args.cluster_info) as f:
        data = json.load(f)

    return _find_local_endpoint(data)


def main():
    parser = argparse.ArgumentParser(description="Setu Client B — pull destination")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Node agent endpoint (tcp://host:port). Overrides auto-detection.",
    )
    parser.add_argument(
        "--cluster-info",
        type=str,
        default=CLUSTER_INFO_FILE,
        help=f"Path to cluster info JSON written by test_script_a.py (default: {CLUSTER_INFO_FILE})",
    )
    args = parser.parse_args()

    endpoint = load_endpoint(args)

    # Connect client to the local node agent
    client = Client()
    print(f"Client B connecting to {endpoint}")
    client.connect(endpoint)

    # Register destination tensor shard on cuda:0
    dst_device = Device(torch_device=torch.device("cuda:1"))
    dims = [
        TensorDimSpec("dim_0", 4, 0, 4),
        TensorDimSpec("dim_1", 8, 0, 8),
    ]
    dst_spec = TensorShardSpec(
        name="pull_dst_tensor",
        dims=dims,
        dtype=torch.float32,
        device=dst_device,
    )
    dst_ref = client.register_tensor_shard(dst_spec)
    assert dst_ref is not None, "Failed to register destination shard"
    print(f"Dest shard registered: {dst_ref.shard_id}")

    client.wait_for_shard_allocation(dst_ref.shard_id)
    print(f"Destination shard allocated, device={dst_device})")

    # Build selection and issue pull
    dim_map = {
        "dim_0": TensorDim("dim_0", 4),
        "dim_1": TensorDim("dim_1", 8),
    }
    src_selection = TensorSelection("pull_src_tensor", dim_map)
    dst_selection = TensorSelection("pull_dst_tensor", dim_map)
    copy_spec = CopySpec(
        "pull_src_tensor",
        "pull_dst_tensor",
        src_selection,
        dst_selection,
    )

    print("Submitting pull: pull_src_tensor -> pull_dst_tensor")
    copy_op_id = client.submit_pull(copy_spec)
    assert copy_op_id is not None, "submit_pull returned None"

    print(f"Waiting for copy (op_id={copy_op_id})...")
    client.wait_for_copy(copy_op_id)
    print("Copy complete!")

    # Verify the pulled data
    ipc_spec, _, _ = client.get_tensor_handle(dst_ref)
    tensor = rebuild_cuda_tensor(
        **ipc_spec.to_dict(),
        tensor_cls=torch.Tensor,
        storage_cls=torch.storage.UntypedStorage,
    )
    torch.cuda.synchronize()

    actual_value = tensor.mean().item()
    print(f"Destination tensor mean: {actual_value} (expected: 42.0)")

    if abs(actual_value - 42.0) < 1e-5:
        print("SUCCESS: Data pulled correctly!")
    else:
        print(f"FAILURE: Expected 42.0, got {actual_value}")
        sys.exit(1)

    client.disconnect()
    print("Client B done.")


if __name__ == "__main__":
    main()
