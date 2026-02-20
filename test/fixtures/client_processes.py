"""
Reusable client process targets for E2E tests.

These are module-level functions that can be used as multiprocessing targets
for source (write) and destination (pull + verify) client processes.
"""

import time

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor


def rebuild_tensor_from_handle(tensor_ipc_spec):
    """Rebuild a CUDA tensor from an IPC spec."""
    spec_dict = tensor_ipc_spec.to_dict()
    return rebuild_cuda_tensor(
        **spec_dict,
        tensor_cls=torch.Tensor,
        storage_cls=torch.storage.UntypedStorage,
    )


def run_source_client(
    client_endpoint,
    tensor_name,
    dims_data,
    init_value,
    init_done_event,
    result_queue,
    client_id,
    device_index=0,
):
    """Source client process: register shard, fill tensor, signal ready.

    Args:
        client_endpoint: ZMQ endpoint to connect to.
        tensor_name: Name of the tensor to register.
        dims_data: List of (name, global_size, start, end) tuples.
        init_value: Value to fill the tensor with.
        init_done_event: Event to set when initialization is complete.
        result_queue: Queue to put result dict into.
        client_id: Identifier for this client (for logging).
        device_index: CUDA device index to use.
    """
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    try:
        print(f"[Source {client_id}] Starting on cuda:{device_index}...", flush=True)
        client = Client()
        client.connect(client_endpoint)

        dims_spec = [
            TensorDimSpec(name, global_size, start, end)
            for name, global_size, start, end in dims_data
        ]
        device = Device(torch_device=torch.device(f"cuda:{device_index}"))
        shard_spec = TensorShardSpec(
            name=tensor_name, dims=dims_spec, dtype=torch.float32, device=device
        )

        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            raise RuntimeError(f"Failed to register source shard {tensor_name}")
        client.wait_for_shard_allocation(shard_ref.shard_id)

        tensor_ipc_spec, _, _ = client.get_tensor_handle(shard_ref)
        tensor = rebuild_tensor_from_handle(tensor_ipc_spec)
        tensor.fill_(init_value)
        torch.cuda.synchronize(device_index)
        print(f"[Source {client_id}] Tensor initialized to {init_value}", flush=True)

        result_queue.put({"success": True, "client_id": client_id})
        init_done_event.set()

        while True:
            time.sleep(0.1)

    except Exception as e:
        import traceback

        print(f"[Source {client_id}] ERROR: {e}", flush=True)
        result_queue.put(
            {
                "success": False,
                "client_id": client_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def run_dest_client(
    client_endpoint,
    src_tensor_name,
    dst_tensor_name,
    dims_data,
    all_sources_ready_event,
    expected_value,
    result_queue,
    client_id,
    device_index=0,
):
    """Dest client process: wait for sources, register shard, pull, verify.

    Args:
        client_endpoint: ZMQ endpoint to connect to.
        src_tensor_name: Name of the source tensor to pull from.
        dst_tensor_name: Name of the destination tensor to register.
        dims_data: List of (name, global_size, start, end) tuples.
        all_sources_ready_event: Event to wait on before starting.
        expected_value: Expected mean value after pull.
        result_queue: Queue to put result dict into.
        client_id: Identifier for this client (for logging).
        device_index: CUDA device index to use.
    """
    from setu._client import Client
    from setu._commons.datatypes import (
        CopySpec,
        Device,
        TensorDim,
        TensorDimSpec,
        TensorSelection,
        TensorShardSpec,
    )

    try:
        print(f"[Dest {client_id}] Waiting for sources...", flush=True)
        if not all_sources_ready_event.wait(timeout=30):
            raise RuntimeError("Timeout waiting for sources")

        client = Client()
        client.connect(client_endpoint)

        dims_spec = [
            TensorDimSpec(name, global_size, start, end)
            for name, global_size, start, end in dims_data
        ]
        device = Device(torch_device=torch.device(f"cuda:{device_index}"))
        shard_spec = TensorShardSpec(
            name=dst_tensor_name, dims=dims_spec, dtype=torch.float32, device=device
        )

        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            raise RuntimeError(f"Failed to register dest shard {dst_tensor_name}")
        client.wait_for_shard_allocation(shard_ref.shard_id)

        dim_map = {
            name: TensorDim(name, global_size) for name, global_size, _, _ in dims_data
        }
        src_selection = TensorSelection(src_tensor_name, dim_map)
        dst_selection = TensorSelection(dst_tensor_name, dim_map)
        copy_spec = CopySpec(
            src_tensor_name, dst_tensor_name, src_selection, dst_selection
        )

        print(f"[Dest {client_id}] Submitting pull...", flush=True)
        copy_op_id = client.submit_pull(copy_spec)
        if copy_op_id is None:
            raise RuntimeError("Failed to submit pull")

        client.wait_for_copy(copy_op_id)
        print(f"[Dest {client_id}] Copy complete!", flush=True)

        tensor_ipc_spec, _, _ = client.get_tensor_handle(shard_ref)
        tensor = rebuild_tensor_from_handle(tensor_ipc_spec)
        torch.cuda.synchronize(device_index)

        actual_value = tensor.mean().item()
        values_match = abs(actual_value - expected_value) < 1e-5
        print(
            f"[Dest {client_id}] expected={expected_value}, "
            f"actual={actual_value}, match={values_match}",
            flush=True,
        )

        result_queue.put(
            {
                "success": True,
                "client_id": client_id,
                "expected_value": expected_value,
                "actual_value": actual_value,
                "values_match": values_match,
            }
        )
        client.disconnect()

    except Exception as e:
        import traceback

        print(f"[Dest {client_id}] ERROR: {e}", flush=True)
        result_queue.put(
            {
                "success": False,
                "client_id": client_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
