"""
Client A: Register a source shard on cuda:0, fill it with 42.0,
then wait for client B to pull.

Assumes the Setu cluster (Coordinator + NodeAgent) is already running.

Usage:
    python -m setu.ray.test_script_a --endpoint tcp://<node_agent_ip>:<port>
"""

import argparse
import signal
import sys
import time

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from setu._client import Client
from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec


def main():
    parser = argparse.ArgumentParser(description="Setu Client A — source shard")
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="Node agent endpoint (tcp://host:port)",
    )
    args = parser.parse_args()

    endpoint = args.endpoint
    print(f"Client A connecting to {endpoint}")

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

    print("Registering source shard...")
    src_ref = client.register_tensor_shard(src_spec)
    assert src_ref is not None, "Failed to register source shard"
    print(f"Source shard registered: {src_ref.shard_id}")

    print("Waiting for shard allocation...")
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

    print("\nClient A ready. Run test_script_b.py on the other node.")
    print("Press Ctrl+C to stop.\n")

    def handle_signal(_signum, _frame):
        print("\nShutting down...")
        client.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Block forever — shard must stay alive for the pull
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
