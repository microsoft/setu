"""
Integration tests for deferred shard deregistration.

Tests that when a client disconnects while copy operations are in-flight,
deregistration is deferred until all blocking copies complete. Specifically:

1. Disconnect with no pending copies → immediate cleanup (no regression)
2. Disconnect while copy is in-flight → deregistration waits for copy to
   complete, then cleans up
3. After deferred deregistration completes, re-registration works
4. Multiple blocking copies → deregistration waits for all
"""

import time
import uuid

import pytest
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor


def _run_coordinator(port: int, ready_event, stop_event):
    """Run the Coordinator in a separate process."""
    from setu._coordinator import Coordinator, NCCLBackend, PassManager, Planner

    backend = NCCLBackend()
    pass_manager = PassManager()
    planner = Planner(backend, pass_manager)

    coordinator = Coordinator(port, planner)
    coordinator.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    coordinator.stop()


def _run_node_agent(
    port: int,
    coordinator_endpoint: str,
    ready_event,
    stop_event,
    node_id=None,
    device_indices=None,
):
    """Run the NodeAgent in a separate process."""
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
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    node_agent.stop()


def _rebuild_tensor_from_handle(tensor_ipc_spec):
    """Rebuild a CUDA tensor from an IPC spec."""
    spec_dict = tensor_ipc_spec.to_dict()
    args = {
        **spec_dict,
        "tensor_cls": torch.Tensor,
        "storage_cls": torch.storage.UntypedStorage,
    }
    return rebuild_cuda_tensor(**args)


# ==============================================================================
# Subprocess helpers for deferred deregistration tests
# ==============================================================================


def _run_source_client_with_disconnect(
    client_endpoint: str,
    tensor_name: str,
    dims_data: list,
    init_value: float,
    init_done_event,
    start_disconnect_event,
    result_queue,
    device_index: int = 0,
):
    """
    Source client process:
    1. Register source shard and initialize data
    2. Signal initialization done
    3. Wait for disconnect signal
    4. Disconnect (blocks until all copies involving our shards complete)
    5. Report timing to result queue

    Args:
        dims_data: List of tuples (name, global_size, start, end).
    """
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    try:
        client = Client()
        client.connect(client_endpoint)

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

        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            raise RuntimeError(f"Failed to register source shard {tensor_name}")

        client.wait_for_shard_allocation(shard_ref.shard_id)

        # Get tensor handle and initialize data
        tensor_ipc_spec, _, _ = client.get_tensor_handle(shard_ref)
        tensor = _rebuild_tensor_from_handle(tensor_ipc_spec)
        tensor.fill_(init_value)
        torch.cuda.synchronize()

        # Signal that initialization is done
        init_done_event.set()

        # Wait for the signal to disconnect
        start_disconnect_event.wait(timeout=60)

        # Disconnect — this sends DeregisterShardsRequest, which will block
        # until all copy operations involving our shards complete
        disconnect_start = time.time()
        client.disconnect()
        disconnect_end = time.time()

        result_queue.put(
            {
                "success": True,
                "disconnect_start": disconnect_start,
                "disconnect_end": disconnect_end,
                "disconnect_duration": disconnect_end - disconnect_start,
            }
        )

    except Exception as e:
        import traceback

        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def _run_dest_client_pull(
    client_endpoint: str,
    src_tensor_name: str,
    dst_tensor_name: str,
    dims_data: list,
    source_ready_event,
    pull_submitted_event,
    expected_value: float,
    result_queue,
    device_index: int = 0,
):
    """
    Destination client process:
    1. Wait for source to be ready
    2. Register destination shard
    3. Submit pull operation
    4. Signal that pull was submitted
    5. Wait for copy to complete and verify data

    Args:
        dims_data: List of tuples (name, global_size, start, end).
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
        # Wait for source data to be initialized
        if not source_ready_event.wait(timeout=30):
            raise RuntimeError("Timeout waiting for source initialization")

        client = Client()
        client.connect(client_endpoint)

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

        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            raise RuntimeError(f"Failed to register dest shard {dst_tensor_name}")

        client.wait_for_shard_allocation(shard_ref.shard_id)

        # Build CopySpec
        dim_map = {}
        for name, global_size, start, end in dims_data:
            dim_map[name] = TensorDim(name, global_size)

        src_selection = TensorSelection(src_tensor_name, dim_map)
        dst_selection = TensorSelection(dst_tensor_name, dim_map)
        copy_spec = CopySpec(
            src_tensor_name, dst_tensor_name, src_selection, dst_selection
        )

        # Submit pull operation
        copy_op_id = client.submit_pull(copy_spec)
        if copy_op_id is None:
            raise RuntimeError("Failed to submit pull operation")

        # Signal that pull was submitted — main test can now trigger disconnect
        if pull_submitted_event is not None:
            pull_submitted_event.set()

        # Wait for copy to complete
        client.wait_for_copy(copy_op_id)

        # Verify data
        tensor_ipc_spec, _, _ = client.get_tensor_handle(shard_ref)
        tensor = _rebuild_tensor_from_handle(tensor_ipc_spec)
        torch.cuda.synchronize()

        actual_value = tensor.mean().item()
        values_match = abs(actual_value - expected_value) < 1e-5

        result_queue.put(
            {
                "success": True,
                "expected_value": expected_value,
                "actual_value": actual_value,
                "values_match": values_match,
            }
        )

        client.disconnect()

    except Exception as e:
        import traceback

        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


# ==============================================================================
# Tests
# ==============================================================================


@pytest.mark.gpu
def test_disconnect_no_pending_copies_immediate():
    """
    Test that disconnect with no pending copies completes immediately.

    This is a regression test to ensure the new deferred deregistration
    logic doesn't add unnecessary delay when there are no blocking copies.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_port = 29700
    node_agent_port = 29701
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    client_endpoint = f"tcp://localhost:{node_agent_port}"

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()

    processes = []

    try:
        # Start infrastructure
        coordinator_proc = ctx.Process(
            target=_run_coordinator,
            args=(coordinator_port, coordinator_ready, stop_event),
        )
        coordinator_proc.start()
        processes.append(coordinator_proc)
        assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

        node_agent_proc = ctx.Process(
            target=_run_node_agent,
            args=(node_agent_port, coordinator_endpoint, node_agent_ready, stop_event),
        )
        node_agent_proc.start()
        processes.append(node_agent_proc)
        assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"

        time.sleep(0.2)

        # Register shards and disconnect immediately (no copies submitted)
        source_init_done = ctx.Event()
        start_disconnect = ctx.Event()
        source_result_queue = ctx.Queue()

        source_proc = ctx.Process(
            target=_run_source_client_with_disconnect,
            args=(
                client_endpoint,
                "no_copy_tensor",
                [("rows", 8, 0, 8), ("cols", 8, 0, 8)],
                1.0,
                source_init_done,
                start_disconnect,
                source_result_queue,
            ),
        )
        source_proc.start()
        processes.append(source_proc)

        # Wait for source to initialize, then immediately signal disconnect
        assert source_init_done.wait(timeout=10), "Source failed to initialize"
        start_disconnect.set()

        # Collect result
        source_result = source_result_queue.get(timeout=10)
        assert source_result[
            "success"
        ], f"Source disconnect failed: {source_result.get('error')}"

        # Disconnect should be fast (< 2 seconds) when no copies are pending
        assert source_result["disconnect_duration"] < 2.0, (
            f"Disconnect took {source_result['disconnect_duration']:.2f}s, "
            f"expected < 2s for no pending copies"
        )

        # Brief delay for cleanup to propagate
        time.sleep(0.3)

        # Verify re-registration works (proves deregistration happened)
        reregister_result = ctx.Queue()
        reregister_proc = ctx.Process(
            target=_verify_reregistration,
            args=(
                client_endpoint,
                "no_copy_tensor",
                [("rows", 8, 0, 8), ("cols", 8, 0, 8)],
                reregister_result,
            ),
        )
        reregister_proc.start()
        processes.append(reregister_proc)

        result = reregister_result.get(timeout=10)
        assert result["success"], f"Re-registration failed: {result.get('error')}"

    finally:
        stop_event.set()
        time.sleep(0.2)
        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()


@pytest.mark.gpu
def test_disconnect_during_active_copy_waits():
    """
    Test that disconnect blocks until in-flight copy operations complete.

    Setup:
    - Source client registers and initializes data
    - Dest client registers and submits pull
    - Source client disconnects while copy may be in-flight
    - Verify: pull completes with correct data, disconnect returns after copy

    This is the core test for deferred deregistration.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_port = 29710
    node_agent_port = 29711
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    client_endpoint = f"tcp://localhost:{node_agent_port}"
    init_value = 42.0

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()

    source_init_done = ctx.Event()
    pull_submitted = ctx.Event()
    start_disconnect = ctx.Event()

    source_result_queue = ctx.Queue()
    dest_result_queue = ctx.Queue()

    processes = []

    try:
        # Start Coordinator
        coordinator_proc = ctx.Process(
            target=_run_coordinator,
            args=(coordinator_port, coordinator_ready, stop_event),
        )
        coordinator_proc.start()
        processes.append(coordinator_proc)
        assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

        # Start NodeAgent
        node_agent_proc = ctx.Process(
            target=_run_node_agent,
            args=(node_agent_port, coordinator_endpoint, node_agent_ready, stop_event),
        )
        node_agent_proc.start()
        processes.append(node_agent_proc)
        assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"

        time.sleep(0.2)

        dims_data = [("rows", 4, 0, 4), ("cols", 4, 0, 4)]

        # Start source client
        source_proc = ctx.Process(
            target=_run_source_client_with_disconnect,
            args=(
                client_endpoint,
                "active_copy_src",
                dims_data,
                init_value,
                source_init_done,
                start_disconnect,
                source_result_queue,
            ),
        )
        source_proc.start()
        processes.append(source_proc)

        # Start dest client
        dest_proc = ctx.Process(
            target=_run_dest_client_pull,
            args=(
                client_endpoint,
                "active_copy_src",
                "active_copy_dst",
                dims_data,
                source_init_done,
                pull_submitted,
                init_value,
                dest_result_queue,
            ),
        )
        dest_proc.start()
        processes.append(dest_proc)

        # Wait for pull to be submitted, then trigger source disconnect
        assert pull_submitted.wait(timeout=30), "Pull was not submitted in time"
        start_disconnect.set()

        # Collect destination result — pull should succeed
        dest_result = dest_result_queue.get(timeout=60)
        assert dest_result["success"], (
            f"Dest client failed: {dest_result.get('error')}\n"
            f"{dest_result.get('traceback', '')}"
        )
        assert dest_result["values_match"], (
            f"Data mismatch: expected {dest_result['expected_value']}, "
            f"got {dest_result['actual_value']}"
        )

        # Collect source result — disconnect should have succeeded
        source_result = source_result_queue.get(timeout=30)
        assert source_result["success"], (
            f"Source disconnect failed: {source_result.get('error')}\n"
            f"{source_result.get('traceback', '')}"
        )

        # Brief delay for cleanup to propagate
        time.sleep(0.3)

        # Verify re-registration works after deferred deregistration
        reregister_result = ctx.Queue()
        reregister_proc = ctx.Process(
            target=_verify_reregistration,
            args=(
                client_endpoint,
                "active_copy_src",
                dims_data,
                reregister_result,
            ),
        )
        reregister_proc.start()
        processes.append(reregister_proc)

        result = reregister_result.get(timeout=10)
        assert result["success"], (
            f"Re-registration after deferred deregistration failed: "
            f"{result.get('error')}"
        )

    finally:
        stop_event.set()
        time.sleep(0.2)
        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()


@pytest.mark.gpu
def test_disconnect_with_multiple_blocking_copies():
    """
    Test that disconnect waits for ALL blocking copies to complete.

    Setup:
    - Source client registers source tensor
    - 2 dest clients each pull into separate dest tensors
    - Source disconnects while both copies may be in-flight
    - Both copies should succeed and source should deregister cleanly
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_port = 29720
    node_agent_port = 29721
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    client_endpoint = f"tcp://localhost:{node_agent_port}"
    init_value = 99.0

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()

    source_init_done = ctx.Event()
    pull_submitted_0 = ctx.Event()
    pull_submitted_1 = ctx.Event()
    start_disconnect = ctx.Event()

    source_result_queue = ctx.Queue()
    dest_result_queue_0 = ctx.Queue()
    dest_result_queue_1 = ctx.Queue()

    processes = []

    try:
        # Start Coordinator
        coordinator_proc = ctx.Process(
            target=_run_coordinator,
            args=(coordinator_port, coordinator_ready, stop_event),
        )
        coordinator_proc.start()
        processes.append(coordinator_proc)
        assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

        # Start NodeAgent
        node_agent_proc = ctx.Process(
            target=_run_node_agent,
            args=(node_agent_port, coordinator_endpoint, node_agent_ready, stop_event),
        )
        node_agent_proc.start()
        processes.append(node_agent_proc)
        assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"

        time.sleep(0.2)

        dims_data = [("rows", 4, 0, 4), ("cols", 4, 0, 4)]

        # Start source client
        source_proc = ctx.Process(
            target=_run_source_client_with_disconnect,
            args=(
                client_endpoint,
                "multi_copy_src",
                dims_data,
                init_value,
                source_init_done,
                start_disconnect,
                source_result_queue,
            ),
        )
        source_proc.start()
        processes.append(source_proc)

        # Start 2 dest clients pulling from the same source
        dest_proc_0 = ctx.Process(
            target=_run_dest_client_pull,
            args=(
                client_endpoint,
                "multi_copy_src",
                "multi_copy_dst_0",
                dims_data,
                source_init_done,
                pull_submitted_0,
                init_value,
                dest_result_queue_0,
            ),
        )
        dest_proc_0.start()
        processes.append(dest_proc_0)

        dest_proc_1 = ctx.Process(
            target=_run_dest_client_pull,
            args=(
                client_endpoint,
                "multi_copy_src",
                "multi_copy_dst_1",
                dims_data,
                source_init_done,
                pull_submitted_1,
                init_value,
                dest_result_queue_1,
            ),
        )
        dest_proc_1.start()
        processes.append(dest_proc_1)

        # Wait for both pulls to be submitted, then trigger disconnect
        assert pull_submitted_0.wait(timeout=30), "Pull 0 not submitted in time"
        assert pull_submitted_1.wait(timeout=30), "Pull 1 not submitted in time"
        start_disconnect.set()

        # Collect results from both destinations
        dest_result_0 = dest_result_queue_0.get(timeout=60)
        assert dest_result_0["success"], (
            f"Dest 0 failed: {dest_result_0.get('error')}\n"
            f"{dest_result_0.get('traceback', '')}"
        )
        assert dest_result_0["values_match"], (
            f"Dest 0 data mismatch: expected {dest_result_0['expected_value']}, "
            f"got {dest_result_0['actual_value']}"
        )

        dest_result_1 = dest_result_queue_1.get(timeout=60)
        assert dest_result_1["success"], (
            f"Dest 1 failed: {dest_result_1.get('error')}\n"
            f"{dest_result_1.get('traceback', '')}"
        )
        assert dest_result_1["values_match"], (
            f"Dest 1 data mismatch: expected {dest_result_1['expected_value']}, "
            f"got {dest_result_1['actual_value']}"
        )

        # Collect source result
        source_result = source_result_queue.get(timeout=30)
        assert source_result["success"], (
            f"Source disconnect failed: {source_result.get('error')}\n"
            f"{source_result.get('traceback', '')}"
        )

        # Verify re-registration
        time.sleep(0.3)
        reregister_result = ctx.Queue()
        reregister_proc = ctx.Process(
            target=_verify_reregistration,
            args=(
                client_endpoint,
                "multi_copy_src",
                dims_data,
                reregister_result,
            ),
        )
        reregister_proc.start()
        processes.append(reregister_proc)

        result = reregister_result.get(timeout=10)
        assert result["success"], (
            f"Re-registration failed after multi-copy deregistration: "
            f"{result.get('error')}"
        )

    finally:
        stop_event.set()
        time.sleep(0.2)
        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()


@pytest.mark.gpu
def test_disconnect_after_copy_completes_allows_reregistration():
    """
    Test that after a copy completes and source disconnects,
    the same tensor ranges can be re-registered by a new client.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_port = 29730
    node_agent_port = 29731
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    client_endpoint = f"tcp://localhost:{node_agent_port}"
    init_value = 7.0

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()

    source_init_done = ctx.Event()
    start_disconnect = ctx.Event()
    source_result_queue = ctx.Queue()
    dest_result_queue = ctx.Queue()

    processes = []

    try:
        # Start infrastructure
        coordinator_proc = ctx.Process(
            target=_run_coordinator,
            args=(coordinator_port, coordinator_ready, stop_event),
        )
        coordinator_proc.start()
        processes.append(coordinator_proc)
        assert coordinator_ready.wait(timeout=10)

        node_agent_proc = ctx.Process(
            target=_run_node_agent,
            args=(node_agent_port, coordinator_endpoint, node_agent_ready, stop_event),
        )
        node_agent_proc.start()
        processes.append(node_agent_proc)
        assert node_agent_ready.wait(timeout=10)

        time.sleep(0.2)

        dims_data = [("rows", 4, 0, 4), ("cols", 4, 0, 4)]

        # Start source
        source_proc = ctx.Process(
            target=_run_source_client_with_disconnect,
            args=(
                client_endpoint,
                "reregister_src",
                dims_data,
                init_value,
                source_init_done,
                start_disconnect,
                source_result_queue,
            ),
        )
        source_proc.start()
        processes.append(source_proc)

        # Start dest — pull_submitted_event is None since we don't need it
        dest_proc = ctx.Process(
            target=_run_dest_client_pull,
            args=(
                client_endpoint,
                "reregister_src",
                "reregister_dst",
                dims_data,
                source_init_done,
                None,  # No pull_submitted event needed
                init_value,
                dest_result_queue,
            ),
        )
        dest_proc.start()
        processes.append(dest_proc)

        # Wait for dest to complete (copy finishes)
        dest_result = dest_result_queue.get(timeout=60)
        assert dest_result["success"], (
            f"Dest failed: {dest_result.get('error')}\n"
            f"{dest_result.get('traceback', '')}"
        )

        # Now disconnect source (copy already completed, should be immediate)
        start_disconnect.set()

        source_result = source_result_queue.get(timeout=10)
        assert source_result[
            "success"
        ], f"Source disconnect failed: {source_result.get('error')}"
        # Should be fast since copy already completed
        assert source_result["disconnect_duration"] < 2.0, (
            f"Disconnect took {source_result['disconnect_duration']:.2f}s, "
            f"expected < 2s when copy already completed"
        )

        time.sleep(0.3)

        # Re-register the same source tensor range
        reregister_result = ctx.Queue()
        reregister_proc = ctx.Process(
            target=_verify_reregistration,
            args=(
                client_endpoint,
                "reregister_src",
                dims_data,
                reregister_result,
            ),
        )
        reregister_proc.start()
        processes.append(reregister_proc)

        result = reregister_result.get(timeout=10)
        assert result["success"], f"Re-registration failed: {result.get('error')}"

    finally:
        stop_event.set()
        time.sleep(0.2)
        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()


@pytest.mark.gpu
def test_registration_rejected_on_partially_deregistered_tensor():
    """
    Test that registering a shard on a partially-deregistered tensor is rejected.

    Setup:
    - Client A registers shard (rows 0-4) for tensor
    - Client B registers shard (rows 4-8) for tensor
    - Client A disconnects (no copies, immediate deregistration)
    - Tensor is now partially freed (has_deregistered_shards=true)
    - Client C tries to register a new shard → should get None (rejected)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_port = 29740
    node_agent_port = 29741
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    client_endpoint = f"tcp://localhost:{node_agent_port}"

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()

    processes = []

    try:
        # Start infrastructure
        coordinator_proc = ctx.Process(
            target=_run_coordinator,
            args=(coordinator_port, coordinator_ready, stop_event),
        )
        coordinator_proc.start()
        processes.append(coordinator_proc)
        assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

        node_agent_proc = ctx.Process(
            target=_run_node_agent,
            args=(node_agent_port, coordinator_endpoint, node_agent_ready, stop_event),
        )
        node_agent_proc.start()
        processes.append(node_agent_proc)
        assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"

        time.sleep(0.2)

        tensor_name = "partial_dereg_tensor"

        # Client A registers rows 0-4, then disconnects
        client_a_init_done = ctx.Event()
        client_a_disconnect = ctx.Event()
        client_a_result = ctx.Queue()

        client_a_proc = ctx.Process(
            target=_run_source_client_with_disconnect,
            args=(
                client_endpoint,
                tensor_name,
                [("rows", 8, 0, 4), ("cols", 8, 0, 8)],
                1.0,
                client_a_init_done,
                client_a_disconnect,
                client_a_result,
            ),
        )
        client_a_proc.start()
        processes.append(client_a_proc)

        # Client B registers rows 4-8 (keeps tensor alive)
        client_b_init_done = ctx.Event()
        client_b_disconnect = ctx.Event()
        client_b_result = ctx.Queue()

        client_b_proc = ctx.Process(
            target=_run_source_client_with_disconnect,
            args=(
                client_endpoint,
                tensor_name,
                [("rows", 8, 4, 8), ("cols", 8, 0, 8)],
                2.0,
                client_b_init_done,
                client_b_disconnect,
                client_b_result,
            ),
        )
        client_b_proc.start()
        processes.append(client_b_proc)

        # Wait for both to initialize
        assert client_a_init_done.wait(timeout=10), "Client A failed to init"
        assert client_b_init_done.wait(timeout=10), "Client B failed to init"

        # Disconnect Client A — no copies, immediate deregistration
        # Tensor is now partially freed (Client B's shards still exist)
        client_a_disconnect.set()
        a_result = client_a_result.get(timeout=10)
        assert a_result[
            "success"
        ], f"Client A disconnect failed: {a_result.get('error')}"

        time.sleep(0.3)

        # Client C tries to register a new shard on the same tensor
        # This should be rejected because the tensor is partially deregistered
        registration_result = ctx.Queue()
        client_c_proc = ctx.Process(
            target=_attempt_registration_expect_rejection,
            args=(
                client_endpoint,
                tensor_name,
                [("rows", 8, 0, 4), ("cols", 8, 0, 8)],
                registration_result,
            ),
        )
        client_c_proc.start()
        processes.append(client_c_proc)

        result = registration_result.get(timeout=10)
        assert result["success"], (
            f"Test failed: {result.get('error')}\n" f"{result.get('traceback', '')}"
        )
        assert result["registration_rejected"], (
            "Registration should have been rejected on partially-deregistered "
            "tensor, but it succeeded"
        )

        # Clean up Client B
        client_b_disconnect.set()
        b_result = client_b_result.get(timeout=10)
        assert b_result[
            "success"
        ], f"Client B disconnect failed: {b_result.get('error')}"

    finally:
        stop_event.set()
        time.sleep(0.2)
        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()


# ==============================================================================
# Helper subprocess functions
# ==============================================================================


def _verify_reregistration(
    client_endpoint: str,
    tensor_name: str,
    dims_data: list,
    result_queue,
    device_index: int = 0,
):
    """
    Verify that a tensor can be re-registered after deregistration.

    Registers the same tensor name with the same dims, then verifies the
    shard_ref is valid. Reports success/failure to result_queue.
    """
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    try:
        client = Client()
        client.connect(client_endpoint)

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

        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            result_queue.put(
                {
                    "success": False,
                    "error": (
                        f"Re-registration returned None for '{tensor_name}' — "
                        f"deregistration may not have completed"
                    ),
                }
            )
        else:
            result_queue.put(
                {
                    "success": True,
                    "shard_id": shard_ref.shard_id,
                }
            )

        client.disconnect()

    except Exception as e:
        import traceback

        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def _attempt_registration_expect_rejection(
    client_endpoint: str,
    tensor_name: str,
    dims_data: list,
    result_queue,
    device_index: int = 0,
):
    """
    Attempt to register a shard and report whether registration was rejected.

    Reports success=True with registration_rejected=True/False indicating
    whether the registration returned None (rejected).
    """
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    try:
        client = Client()
        client.connect(client_endpoint)

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

        shard_ref = client.register_tensor_shard(shard_spec)

        result_queue.put(
            {
                "success": True,
                "registration_rejected": shard_ref is None,
            }
        )

        client.disconnect()

    except Exception as e:
        import traceback

        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pytest.main([__file__, "-v", "-s", "--gpu"])
