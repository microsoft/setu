"""
Unit tests for Client-NodeAgent interaction.

Tests the communication flow between Client and NodeAgent components.
"""

import time
import traceback
import uuid

import pytest
import torch
import torch.multiprocessing as mp


def _run_coordinator(
    router_executor_port: int,
    router_handler_port: int,
    ready_event,
    stop_event,
):
    """Run the Coordinator in a separate process."""
    from setu._coordinator import Coordinator

    coordinator = Coordinator(router_executor_port, router_handler_port)
    coordinator.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    coordinator.stop()


def _run_node_agent(
    router_port: int,
    dealer_executor_port: int,
    dealer_handler_port: int,
    ready_event,
    stop_event,
):
    """Run the NodeAgent in a separate process."""
    from setu._commons.datatypes import Device
    from setu._node_manager import NodeAgent

    node_id = uuid.uuid4()
    devices = [
        Device(
            node_id=node_id,
            device_rank=0,
            torch_device=torch.device("cuda:0"),
        )
    ]

    node_agent = NodeAgent(
        node_id=node_id,
        router_port=router_port,
        dealer_executor_port=dealer_executor_port,
        dealer_handler_port=dealer_handler_port,
        devices=devices,
    )
    node_agent.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    node_agent.stop()


def _run_client_register_tensor(endpoint: str, tensor_name: str, result_queue):
    """Client process that registers a tensor shard."""
    try:
        from setu._client import Client
        from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

        client = Client()
        client.connect(endpoint)

        device = Device(
            node_id=uuid.uuid4(),
            device_rank=0,
            torch_device=torch.device("cuda:0"),
        )
        dims = [
            TensorDimSpec("dim_0", 32, 0, 32),
            TensorDimSpec("dim_1", 64, 0, 64),
        ]
        shard_spec = TensorShardSpec(
            name=tensor_name,
            dims=dims,
            dtype=torch.float32,
            device=device,
        )

        shard_ref = client.register_tensor_shard(shard_spec)
        client.disconnect()

        result_queue.put(
            {
                "success": True,
                "has_shard_ref": shard_ref is not None,
                "tensor_name": tensor_name,
            }
        )

    except Exception as e:
        traceback.print_exc()
        result_queue.put({"success": False, "error": str(e)})


def _run_client_get_handle(endpoint: str, tensor_name: str, result_queue):
    """Client process that registers a tensor and gets its IPC handle."""
    try:
        from setu._client import Client
        from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

        client = Client()
        client.connect(endpoint)

        device = Device(
            node_id=uuid.uuid4(),
            device_rank=0,
            torch_device=torch.device("cuda:0"),
        )
        dims = [
            TensorDimSpec("dim_0", 4, 0, 4),
            TensorDimSpec("dim_1", 8, 0, 8),
        ]
        shard_spec = TensorShardSpec(
            name=tensor_name,
            dims=dims,
            dtype=torch.float32,
            device=device,
        )

        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            result_queue.put({"success": False, "error": "Failed to register tensor"})
            client.disconnect()
            return

        tensor_ipc_spec = client.get_tensor_handle(tensor_name)
        spec_dict = tensor_ipc_spec.to_dict()
        client.disconnect()

        result_queue.put(
            {
                "success": True,
                "spec_dict": spec_dict,
                "tensor_size": spec_dict.get("tensor_size"),
            }
        )

    except Exception as e:
        traceback.print_exc()
        result_queue.put({"success": False, "error": str(e)})


@pytest.mark.gpu
def test_register_tensor_shard():
    """Test registering a tensor shard through Client -> NodeAgent -> Coordinator."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Ports
    coordinator_executor_port = 29000
    coordinator_handler_port = 29001
    node_agent_router_port = 29100
    endpoint = f"tcp://localhost:{node_agent_router_port}"

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()
    result_queue = ctx.Queue()

    # Start Coordinator
    coordinator_proc = ctx.Process(
        target=_run_coordinator,
        args=(
            coordinator_executor_port,
            coordinator_handler_port,
            coordinator_ready,
            stop_event,
        ),
    )
    coordinator_proc.start()
    assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

    # Start NodeAgent
    node_agent_proc = ctx.Process(
        target=_run_node_agent,
        args=(
            node_agent_router_port,
            coordinator_executor_port,
            coordinator_handler_port,
            node_agent_ready,
            stop_event,
        ),
    )
    node_agent_proc.start()
    assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"

    # Give infrastructure time to fully initialize
    time.sleep(0.5)

    try:
        # Run client
        client_proc = ctx.Process(
            target=_run_client_register_tensor,
            args=(endpoint, "test_tensor", result_queue),
        )
        client_proc.start()
        client_proc.join(timeout=15)

        assert (
            client_proc.exitcode == 0
        ), f"Client process failed: {client_proc.exitcode}"

        result = result_queue.get(timeout=5)
        assert result["success"], f"Client error: {result.get('error')}"
        assert result["has_shard_ref"], "Should receive a valid TensorShardRef"
        assert result["tensor_name"] == "test_tensor"

    finally:
        # Cleanup
        stop_event.set()
        time.sleep(0.3)
        for proc in [node_agent_proc, coordinator_proc]:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


@pytest.mark.gpu
def test_get_tensor_handle():
    """Test getting a tensor IPC handle after registration."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Ports (different from other test to avoid conflicts)
    coordinator_executor_port = 29010
    coordinator_handler_port = 29011
    node_agent_router_port = 29110
    endpoint = f"tcp://localhost:{node_agent_router_port}"

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()
    result_queue = ctx.Queue()

    # Start Coordinator
    coordinator_proc = ctx.Process(
        target=_run_coordinator,
        args=(
            coordinator_executor_port,
            coordinator_handler_port,
            coordinator_ready,
            stop_event,
        ),
    )
    coordinator_proc.start()
    assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

    # Start NodeAgent
    node_agent_proc = ctx.Process(
        target=_run_node_agent,
        args=(
            node_agent_router_port,
            coordinator_executor_port,
            coordinator_handler_port,
            node_agent_ready,
            stop_event,
        ),
    )
    node_agent_proc.start()
    assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"

    time.sleep(0.5)

    try:
        # Run client
        client_proc = ctx.Process(
            target=_run_client_get_handle,
            args=(endpoint, "handle_test_tensor", result_queue),
        )
        client_proc.start()
        client_proc.join(timeout=20)

        assert (
            client_proc.exitcode == 0
        ), f"Client process failed: {client_proc.exitcode}"

        result = result_queue.get(timeout=5)
        assert result["success"], f"Client error: {result.get('error')}"
        assert result["tensor_size"] == [
            4,
            8,
        ], f"Unexpected tensor size: {result['tensor_size']}"

        # Verify spec contains required fields
        spec_dict = result["spec_dict"]
        required_fields = [
            "tensor_size",
            "tensor_stride",
            "storage_handle",
            "storage_size_bytes",
            "ref_counter_handle",
            "event_handle",
        ]
        for field in required_fields:
            assert field in spec_dict, f"Missing field: {field}"

    finally:
        stop_event.set()
        time.sleep(0.3)
        for proc in [node_agent_proc, coordinator_proc]:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


@pytest.mark.gpu
def test_multiple_tensor_registrations():
    """Test registering multiple tensors from a single client."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_executor_port = 29020
    coordinator_handler_port = 29021
    node_agent_router_port = 29120
    endpoint = f"tcp://localhost:{node_agent_router_port}"

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()
    result_queue = ctx.Queue()

    coordinator_proc = ctx.Process(
        target=_run_coordinator,
        args=(
            coordinator_executor_port,
            coordinator_handler_port,
            coordinator_ready,
            stop_event,
        ),
    )
    coordinator_proc.start()
    assert coordinator_ready.wait(timeout=10)

    node_agent_proc = ctx.Process(
        target=_run_node_agent,
        args=(
            node_agent_router_port,
            coordinator_executor_port,
            coordinator_handler_port,
            node_agent_ready,
            stop_event,
        ),
    )
    node_agent_proc.start()
    assert node_agent_ready.wait(timeout=10)

    time.sleep(0.5)

    try:
        # Register 3 tensors sequentially
        for i in range(3):
            client_proc = ctx.Process(
                target=_run_client_register_tensor,
                args=(endpoint, f"tensor_{i}", result_queue),
            )
            client_proc.start()
            client_proc.join(timeout=15)

            assert client_proc.exitcode == 0
            result = result_queue.get(timeout=5)
            assert result["success"], f"Tensor {i} failed: {result.get('error')}"
            assert result["has_shard_ref"]

    finally:
        stop_event.set()
        time.sleep(0.3)
        for proc in [node_agent_proc, coordinator_proc]:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pytest.main([__file__, "-v", "--gpu"])
