"""
Test multi-client pull operation with separate processes.

Tests the flow where:
- 2 source clients register source shards and initialize data
- 4 destination clients register destination shards and pull data
"""

import os
import time
import uuid

import pytest
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor

# Enable debug logging for all processes
os.environ["SETU_LOG_LEVEL"] = "DEBUG"


def _run_coordinator(port: int, ready_event, stop_event):
    """Run the Coordinator in a separate process."""
    print(f"[Coordinator] Starting on port {port}...", flush=True)
    from setu._coordinator import Coordinator

    coordinator = Coordinator(port)
    coordinator.start()
    print(f"[Coordinator] Started successfully", flush=True)
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    print(f"[Coordinator] Stopping...", flush=True)
    coordinator.stop()
    print(f"[Coordinator] Stopped", flush=True)


def _run_node_agent(
    port: int,
    coordinator_endpoint: str,
    ready_event,
    stop_event,
    node_id=None,
    device_indices: list = None,
):
    """Run the NodeAgent in a separate process."""
    print(f"[NodeAgent] Starting on port {port}...", flush=True)
    from setu._commons.datatypes import Device
    from setu._node_manager import NodeAgent

    if node_id is None:
        node_id = uuid.uuid4()

    if device_indices is None:
        device_indices = [0]

    devices = [Device(torch_device=torch.device(f"cuda:{i}")) for i in device_indices]

    node_agent = NodeAgent(
        node_id=node_id,
        port=port,
        coordinator_endpoint=coordinator_endpoint,
        devices=devices,
    )
    node_agent.start()
    print(f"[NodeAgent] Started successfully", flush=True)
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    print(f"[NodeAgent] Stopping...", flush=True)
    node_agent.stop()
    print(f"[NodeAgent] Stopped", flush=True)


def _rebuild_tensor_from_handle(tensor_ipc_spec):
    """Rebuild a CUDA tensor from an IPC spec."""
    spec_dict = tensor_ipc_spec.to_dict()
    args = {
        **spec_dict,
        "tensor_cls": torch.Tensor,
        "storage_cls": torch.storage.UntypedStorage,
    }
    return rebuild_cuda_tensor(**args)


def _run_source_client(
    client_endpoint: str,
    tensor_name: str,
    dims_data: list,
    init_value: float,
    init_done_event,
    result_queue,
    client_id: int,
    device_index: int = 0,
):
    """
    Source client process:
    1. Register source shard
    2. Get tensor handle and initialize to init_value
    3. Signal that initialization is done

    Args:
        dims_data: List of tuples (name, global_size, start, end) - picklable raw data
    """
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    try:
        print(f"[Source {client_id}] Starting for {tensor_name} with dims_data {dims_data}...", flush=True)
        client = Client()
        client.connect(client_endpoint)
        print(f"[Source {client_id}] Connected to {client_endpoint}", flush=True)

        # Create TensorDimSpec objects inside the process
        dims_spec = [
            TensorDimSpec(name, global_size, start, end)
            for name, global_size, start, end in dims_data
        ]

        device = Device(torch_device=torch.device(f"cuda:{device_index}"))
        shard_spec = TensorShardSpec(
            name=tensor_name,
            dims=dims_spec,
            dtype=torch.float32,
            device=device,
        )

        print(f"[Source {client_id}] Registering shard...", flush=True)
        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            raise RuntimeError(f"Failed to register source shard {tensor_name}")
        print(f"[Source {client_id}] Shard registered: {shard_ref.shard_id} for tensor '{tensor_name}'", flush=True)

        # Get tensor handle and initialize data
        print(f"[Source {client_id}] Getting tensor handle...", flush=True)
        tensor_ipc_spec = client.get_tensor_handle(shard_ref)
        tensor = _rebuild_tensor_from_handle(tensor_ipc_spec)
        print(f"[Source {client_id}] Filling tensor with {init_value}...", flush=True)
        tensor.fill_(init_value)
        torch.cuda.synchronize()
        print(f"[Source {client_id}] Tensor initialized!", flush=True)

        result_queue.put({
            "success": True,
            "client_id": client_id,
            "tensor_name": tensor_name,
            "shard_id": shard_ref.shard_id,
        })

        # Signal that this source is initialized
        init_done_event.set()
        print(f"[Source {client_id}] Signaled ready, waiting...", flush=True)

        # Keep client alive until test completes (wait for stop signal implicitly)
        # The process will be terminated by the test fixture
        while True:
            time.sleep(0.1)

    except Exception as e:
        import traceback
        print(f"[Source {client_id}] ERROR: {e}", flush=True)
        result_queue.put({
            "success": False,
            "client_id": client_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


def _run_dest_client(
    client_endpoint: str,
    src_tensor_name: str,
    dst_tensor_name: str,
    dims_data: list,
    all_sources_ready_event,
    expected_value: float,
    result_queue,
    client_id: int,
    device_index: int = 0,
):
    """
    Destination client process:
    1. Wait for all sources to be initialized
    2. Register destination shard
    3. Submit pull operation
    4. Get tensor handle and verify value

    Args:
        dims_data: List of tuples (name, global_size, start, end) - picklable raw data
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
        print(f"[Dest {client_id}] Starting for {dst_tensor_name}... with dims_data {dims_data}", flush=True)
        # Wait for all sources to be ready
        print(f"[Dest {client_id}] Waiting for sources...", flush=True)
        if not all_sources_ready_event.wait(timeout=30):
            raise RuntimeError("Timeout waiting for sources to initialize")
        print(f"[Dest {client_id}] Sources ready!", flush=True)

        client = Client()
        client.connect(client_endpoint)
        print(f"[Dest {client_id}] Connected to {client_endpoint}", flush=True)

        # Create TensorDimSpec objects inside the process
        dims_spec = [
            TensorDimSpec(name, global_size, start, end)
            for name, global_size, start, end in dims_data
        ]

        device = Device(torch_device=torch.device(f"cuda:{device_index}"))
        shard_spec = TensorShardSpec(
            name=dst_tensor_name,
            dims=dims_spec,
            dtype=torch.float32,
            device=device,
        )

        print(f"[Dest {client_id}] Registering shard...", flush=True)
        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            raise RuntimeError(f"Failed to register dest shard {dst_tensor_name}")
        print(f"[Dest {client_id}] Shard registered: {shard_ref.shard_id}", flush=True)

        # Build CopySpec for pull operation
        # Create dimension map for the full tensor selection
        dim_map = {}
        for name, global_size, start, end in dims_data:
            dim_map[name] = TensorDim(name, global_size)

        src_selection = TensorSelection(src_tensor_name, dim_map)
        dst_selection = TensorSelection(dst_tensor_name, dim_map)
        copy_spec = CopySpec(
            src_tensor_name, dst_tensor_name, src_selection, dst_selection
        )

        time.sleep(2)

        # Submit pull operation
        print(f"[Dest {client_id}] Submitting pull from {src_tensor_name}...", flush=True)
        copy_op_id = client.submit_pull(copy_spec)
        if copy_op_id is None:
            raise RuntimeError("Failed to submit pull operation")
        print(f"[Dest {client_id}] Pull submitted, copy_op_id={copy_op_id}", flush=True)

        # Wait for copy to complete
        print(f"[Dest {client_id}] Waiting for copy to complete...", flush=True)
        client.wait_for_copy(copy_op_id)
        print(f"[Dest {client_id}] Copy complete!", flush=True)

        # Get tensor handle and verify value
        print(f"[Dest {client_id}] Getting tensor handle...", flush=True)
        tensor_ipc_spec = client.get_tensor_handle(shard_ref)
        tensor = _rebuild_tensor_from_handle(tensor_ipc_spec)
        torch.cuda.synchronize()

        actual_value = tensor.mean().item()
        values_match = abs(actual_value - expected_value) < 1e-5
        print(f"[Dest {client_id}] Value check: expected={expected_value}, actual={actual_value}, match={values_match}", flush=True)

        result_queue.put({
            "success": True,
            "client_id": client_id,
            "tensor_name": dst_tensor_name,
            "expected_value": expected_value,
            "actual_value": actual_value,
            "values_match": values_match,
        })

        client.disconnect()
        print(f"[Dest {client_id}] Done!", flush=True)

    except Exception as e:
        import traceback
        print(f"[Dest {client_id}] ERROR: {e}", flush=True)
        result_queue.put({
            "success": False,
            "client_id": client_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


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

    # Configuration
    coordinator_port = 29500
    node_agent_port = 29600
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    client_endpoint = f"tcp://localhost:{node_agent_port}"
    init_value = 10.0

    dim_names = ["a", "b"]
    dim_sizes = [4,4]

    def make_dims_data(dim_names, dim_sizes, num_shards, i, shard_dim=0):
        dim_owned_range = []
        for idx, sz in enumerate(dim_sizes):
            if idx == shard_dim:
                shard_sz = sz // num_shards
                dim_owned_range.append((i * shard_sz, (i + 1) * shard_sz))
            else:
                dim_owned_range.append((0, sz))
        return [(n, sz, s, e) for (n, sz, (s, e)) in zip(dim_names, dim_sizes, dim_owned_range)]
        
    ctx = mp.get_context("spawn")

    # Events
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()
    source_init_events = [ctx.Event() for _ in range(2)]
    all_sources_ready = ctx.Event()

    # Result queues
    source_results = ctx.Queue()
    dest_results = ctx.Queue()

    processes = []

    try:
        # Start Coordinator
        print("[Main] Starting Coordinator...", flush=True)
        coordinator_proc = ctx.Process(
            target=_run_coordinator,
            args=(coordinator_port, coordinator_ready, stop_event),
        )
        coordinator_proc.start()
        processes.append(coordinator_proc)
        assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"
        print("[Main] Coordinator ready!", flush=True)

        # Start NodeAgent
        print("[Main] Starting NodeAgent...", flush=True)
        node_agent_proc = ctx.Process(
            target=_run_node_agent,
            args=(node_agent_port, coordinator_endpoint, node_agent_ready, stop_event),
            kwargs={"device_indices": [0]},
        )
        node_agent_proc.start()
        processes.append(node_agent_proc)
        assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"
        print("[Main] NodeAgent ready!", flush=True)

        time.sleep(0.2)  # Brief delay for initialization

        # Start 2 source clients
        print("[Main] Starting 2 source clients...", flush=True)
        for i in range(2):
            proc = ctx.Process(
                target=_run_source_client,
                args=(
                    client_endpoint,
                    f"source_tensor",
                    make_dims_data(dim_names, dim_sizes, 2, i),
                    init_value,
                    source_init_events[i],
                    source_results,
                    i,
                ),
            )
            proc.start()
            processes.append(proc)

        # Wait for all sources to initialize
        print("[Main] Waiting for source clients to initialize...", flush=True)
        for i, event in enumerate(source_init_events):
            assert event.wait(timeout=30), f"Source client {i} failed to initialize"
            print(f"[Main] Source client {i} initialized!", flush=True)

        # Signal that all sources are ready
        print("[Main] All sources ready, signaling dest clients...", flush=True)
        all_sources_ready.set()

        # Start 4 destination clients (2 per source)
        print("[Main] Starting 4 destination clients...", flush=True)
        for i in range(4):
            proc = ctx.Process(
                target=_run_dest_client,
                args=(
                    client_endpoint,
                    f"source_tensor",
                    f"dest_tensor",
                    make_dims_data(dim_names, dim_sizes, 4, i),
                    all_sources_ready,
                    init_value,
                    dest_results,
                    i,
                ),
            )
            proc.start()
            processes.append(proc)

        # Collect source results
        print("[Main] Collecting source results...", flush=True)
        source_client_results = []
        for _ in range(2):
            result = source_results.get(timeout=30)
            source_client_results.append(result)
            assert result["success"], f"Source client failed: {result.get('error')}"
        print("[Main] All source results collected!", flush=True)

        # Collect destination results
        print("[Main] Collecting destination results...", flush=True)
        dest_client_results = []
        for _ in range(4):
            result = dest_results.get(timeout=60)
            print(f"[Main] Got dest result: {result}", flush=True)
            dest_client_results.append(result)

        # Verify all destination clients succeeded
        for result in dest_client_results:
            assert result["success"], (
                f"Dest client {result.get('client_id')} failed: "
                f"{result.get('error')}\n{result.get('traceback', '')}"
            )
            assert result["values_match"], (
                f"Dest client {result['client_id']}: "
                f"expected {result['expected_value']}, got {result['actual_value']}"
            )

    finally:
        # Cleanup
        stop_event.set()
        time.sleep(0.2)

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

    # Configuration
    coordinator_port = 29501
    node_agent_port = 29601
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    client_endpoint = f"tcp://localhost:{node_agent_port}"
    init_value = 10.0

    dim_names = ["a", "b"]
    dim_sizes = [4, 4]

    def make_dims_data(dim_names, dim_sizes, num_shards, i, shard_dim=0):
        dim_owned_range = []
        for idx, sz in enumerate(dim_sizes):
            if idx == shard_dim:
                shard_sz = sz // num_shards
                dim_owned_range.append((i * shard_sz, (i + 1) * shard_sz))
            else:
                dim_owned_range.append((0, sz))
        return [(n, sz, s, e) for (n, sz, (s, e)) in zip(dim_names, dim_sizes, dim_owned_range)]

    ctx = mp.get_context("spawn")

    # Events
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()
    source_init_events = [ctx.Event() for _ in range(2)]
    all_sources_ready = ctx.Event()

    # Result queues
    source_results = ctx.Queue()
    dest_results = ctx.Queue()

    processes = []

    try:
        # Start Coordinator
        print("[Main-MD] Starting Coordinator...", flush=True)
        coordinator_proc = ctx.Process(
            target=_run_coordinator,
            args=(coordinator_port, coordinator_ready, stop_event),
        )
        coordinator_proc.start()
        processes.append(coordinator_proc)
        assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"
        print("[Main-MD] Coordinator ready!", flush=True)

        # Start NodeAgent with 4 devices
        print("[Main-MD] Starting NodeAgent with devices [0,1,2,3]...", flush=True)
        node_agent_proc = ctx.Process(
            target=_run_node_agent,
            args=(node_agent_port, coordinator_endpoint, node_agent_ready, stop_event),
            kwargs={"device_indices": [0, 1, 2, 3]},
        )
        node_agent_proc.start()
        processes.append(node_agent_proc)
        assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"
        print("[Main-MD] NodeAgent ready!", flush=True)

        time.sleep(0.2)  # Brief delay for initialization

        # Start 2 source clients on different devices
        print("[Main-MD] Starting 2 source clients on cuda:0 and cuda:1...", flush=True)
        for i in range(2):
            proc = ctx.Process(
                target=_run_source_client,
                args=(
                    client_endpoint,
                    f"source_tensor",
                    make_dims_data(dim_names, dim_sizes, 2, i),
                    init_value,
                    source_init_events[i],
                    source_results,
                    i,
                ),
                kwargs={"device_index": i},
            )
            proc.start()
            processes.append(proc)

        # Wait for all sources to initialize
        print("[Main-MD] Waiting for source clients to initialize...", flush=True)
        for i, event in enumerate(source_init_events):
            assert event.wait(timeout=30), f"Source client {i} failed to initialize"
            print(f"[Main-MD] Source client {i} initialized!", flush=True)

        # Signal that all sources are ready
        print("[Main-MD] All sources ready, signaling dest clients...", flush=True)
        all_sources_ready.set()

        # Start 4 destination clients, each on a different device
        print("[Main-MD] Starting 4 destination clients on cuda:0..3...", flush=True)
        for i in range(4):
            proc = ctx.Process(
                target=_run_dest_client,
                args=(
                    client_endpoint,
                    f"source_tensor",
                    f"dest_tensor",
                    make_dims_data(dim_names, dim_sizes, 4, i),
                    all_sources_ready,
                    init_value,
                    dest_results,
                    i,
                ),
                kwargs={"device_index": i},
            )
            proc.start()
            processes.append(proc)

        # Collect source results
        print("[Main-MD] Collecting source results...", flush=True)
        source_client_results = []
        for _ in range(2):
            result = source_results.get(timeout=30)
            source_client_results.append(result)
            assert result["success"], f"Source client failed: {result.get('error')}"
        print("[Main-MD] All source results collected!", flush=True)

        # Collect destination results
        print("[Main-MD] Collecting destination results...", flush=True)
        dest_client_results = []
        for _ in range(4):
            result = dest_results.get(timeout=60)
            print(f"[Main-MD] Got dest result: {result}", flush=True)
            dest_client_results.append(result)

        # Verify all destination clients succeeded
        for result in dest_client_results:
            assert result["success"], (
                f"Dest client {result.get('client_id')} failed: "
                f"{result.get('error')}\n{result.get('traceback', '')}"
            )
            assert result["values_match"], (
                f"Dest client {result['client_id']}: "
                f"expected {result['expected_value']}, got {result['actual_value']}"
            )

    finally:
        # Cleanup
        stop_event.set()
        time.sleep(0.2)

        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
