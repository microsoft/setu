"""
Unit tests for Client-NodeAgent interaction.

Tests the communication flow between Client and NodeAgent components.
"""

import time
import uuid

import pytest
import torch
import torch.multiprocessing as mp


def _run_coordinator(
    port: int,
    ready_event,
    stop_event,
):
    """Run the Coordinator in a separate process."""
    from setu._coordinator import Coordinator

    coordinator = Coordinator(port)
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
    device_rank: int = 0,
    device_index: int = 0,
):
    """Run the NodeAgent in a separate process."""
    from setu._commons.datatypes import Device
    from setu._node_manager import NodeAgent

    if node_id is None:
        node_id = uuid.uuid4()

    devices = [
        Device(
            node_id=node_id,
            device_rank=device_rank,
            torch_device=torch.device(f"cuda:{device_index}"),
        )
    ]

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


def _register_tensor(
    endpoint: str,
    tensor_name: str,
    dims_spec=None,
    node_id=None,
    device_rank: int = 0,
    device_index: int = 0,
):
    """Register a tensor shard and return the shard ref."""
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    if dims_spec is None:
        dims_spec = [
            TensorDimSpec("dim_0", 32, 0, 32),
            TensorDimSpec("dim_1", 64, 0, 64),
        ]

    if node_id is None:
        node_id = uuid.uuid4()

    client = Client()
    client.connect(endpoint)

    device = Device(
        node_id=node_id,
        device_rank=device_rank,
        torch_device=torch.device(f"cuda:{device_index}"),
    )
    shard_spec = TensorShardSpec(
        name=tensor_name,
        dims=dims_spec,
        dtype=torch.float32,
        device=device,
    )

    shard_ref = client.register_tensor_shard(shard_spec)
    client.disconnect()

    return shard_ref


def _register_and_get_handle(endpoint: str, tensor_name: str, dims_spec):
    """Register a tensor and get its IPC handle."""
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorShardSpec

    client = Client()
    client.connect(endpoint)

    device = Device(
        node_id=uuid.uuid4(),
        device_rank=0,
        torch_device=torch.device("cuda:0"),
    )
    shard_spec = TensorShardSpec(
        name=tensor_name,
        dims=dims_spec,
        dtype=torch.float32,
        device=device,
    )

    shard_ref = client.register_tensor_shard(shard_spec)
    if shard_ref is None:
        raise RuntimeError("Failed to register tensor")

    tensor_ipc_spec = client.get_tensor_handle(tensor_name)
    client.disconnect()

    return tensor_ipc_spec


@pytest.fixture(scope="module")
def infrastructure():
    """
    Start Coordinator and NodeAgent once for all tests in this module.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_port = 29000
    node_agent_port = 29100
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    client_endpoint = f"tcp://localhost:{node_agent_port}"

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()

    # Start Coordinator
    coordinator_proc = ctx.Process(
        target=_run_coordinator,
        args=(
            coordinator_port,
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
            node_agent_port,
            coordinator_endpoint,
            node_agent_ready,
            stop_event,
        ),
    )
    node_agent_proc.start()
    assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"

    # Brief delay for initialization
    time.sleep(0.1)

    # Yield infrastructure for tests to use
    yield {
        "client_endpoint": client_endpoint,
    }

    # Cleanup
    stop_event.set()
    time.sleep(0.1)
    for proc in [node_agent_proc, coordinator_proc]:
        proc.join(timeout=3)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)


@pytest.mark.gpu
def test_register_tensor_shard(infrastructure):
    """Test registering a tensor shard through Client -> NodeAgent -> Coordinator."""
    client_endpoint = infrastructure["client_endpoint"]

    shard_ref = _register_tensor(client_endpoint, "test_tensor")

    assert shard_ref is not None, "Should receive a valid TensorShardRef"
    assert shard_ref.name == "test_tensor"


@pytest.mark.gpu
def test_get_tensor_handle(infrastructure):
    """Test getting a tensor IPC handle after registration."""
    from setu._commons.datatypes import TensorDimSpec

    client_endpoint = infrastructure["client_endpoint"]

    dims = [
        TensorDimSpec("dim_0", 4, 0, 4),
        TensorDimSpec("dim_1", 8, 0, 8),
    ]

    tensor_ipc_spec = _register_and_get_handle(
        client_endpoint, "handle_test_tensor", dims
    )
    spec_dict = tensor_ipc_spec.to_dict()

    assert spec_dict.get("tensor_size") == [
        4,
        8,
    ], f"Unexpected tensor size: {spec_dict.get('tensor_size')}"

    # Verify spec contains required fields
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


@pytest.mark.gpu
def test_multiple_tensor_registrations(infrastructure):
    """Test registering multiple tensors from a single client."""
    client_endpoint = infrastructure["client_endpoint"]

    # Register 3 tensors sequentially
    for i in range(3):
        shard_ref = _register_tensor(client_endpoint, f"tensor_{i}")
        assert shard_ref is not None, f"Failed to register tensor_{i}"
        assert shard_ref.name == f"tensor_{i}"


@pytest.fixture(scope="function")
def multi_node_infrastructure():
    """Start Coordinator and two NodeAgents for distributed tensor tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_port = 29200
    node_agent_0_port = 29201
    node_agent_1_port = 29202
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"

    node_id_0 = uuid.uuid4()
    node_id_1 = uuid.uuid4()

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_0_ready = ctx.Event()
    node_agent_1_ready = ctx.Event()
    stop_event = ctx.Event()

    # Start Coordinator
    coordinator_proc = ctx.Process(
        target=_run_coordinator,
        args=(coordinator_port, coordinator_ready, stop_event),
    )
    coordinator_proc.start()
    assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

    # Start NodeAgent 0 (GPU 0)
    node_agent_0_proc = ctx.Process(
        target=_run_node_agent,
        args=(
            node_agent_0_port,
            coordinator_endpoint,
            node_agent_0_ready,
            stop_event,
            node_id_0,
            0,
            0,
        ),
    )
    node_agent_0_proc.start()
    assert node_agent_0_ready.wait(timeout=10), "NodeAgent 0 failed to start"

    # Start NodeAgent 1 (GPU 1)
    node_agent_1_proc = ctx.Process(
        target=_run_node_agent,
        args=(
            node_agent_1_port,
            coordinator_endpoint,
            node_agent_1_ready,
            stop_event,
            node_id_1,
            1,
            1,
        ),
    )
    node_agent_1_proc.start()
    assert node_agent_1_ready.wait(timeout=10), "NodeAgent 1 failed to start"

    time.sleep(0.2)

    yield {
        "client_endpoint_0": f"tcp://localhost:{node_agent_0_port}",
        "client_endpoint_1": f"tcp://localhost:{node_agent_1_port}",
        "node_id_0": node_id_0,
        "node_id_1": node_id_1,
    }

    stop_event.set()
    time.sleep(0.1)
    for proc in [node_agent_0_proc, node_agent_1_proc, coordinator_proc]:
        proc.join(timeout=3)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)


@pytest.mark.gpu
def test_distributed_tensor_allocation(multi_node_infrastructure):
    """
    Test that AllocateTensorRequest is broadcast to ALL NodeAgents
    when a tensor is distributed across multiple nodes.
    """
    from setu._client import Client
    from setu._commons.datatypes import TensorDimSpec

    infra = multi_node_infrastructure
    tensor_name = "distributed_tensor"
    total_rows = 1024

    # Node 0 owns rows [0, 512)
    dims_0 = [
        TensorDimSpec("rows", total_rows, 0, 512),
        TensorDimSpec("cols", 768, 0, 768),
    ]
    shard_ref_0 = _register_tensor(
        infra["client_endpoint_0"],
        tensor_name,
        dims_0,
        infra["node_id_0"],
        0,
        0,
    )
    assert shard_ref_0 is not None

    # Node 1 owns rows [512, 1024)
    dims_1 = [
        TensorDimSpec("rows", total_rows, 512, 1024),
        TensorDimSpec("cols", 768, 0, 768),
    ]
    shard_ref_1 = _register_tensor(
        infra["client_endpoint_1"],
        tensor_name,
        dims_1,
        infra["node_id_1"],
        1,
        1,
    )
    assert shard_ref_1 is not None

    # Wait for AllocateTensorRequest to be processed
    time.sleep(0.5)

    # Verify both NodeAgents allocated the tensor by getting handles
    client_0 = Client()
    client_0.connect(infra["client_endpoint_0"])
    handle_0 = client_0.get_tensor_handle(tensor_name)
    client_0.disconnect()

    client_1 = Client()
    client_1.connect(infra["client_endpoint_1"])
    handle_1 = client_1.get_tensor_handle(tensor_name)
    client_1.disconnect()

    assert handle_0 is not None, "NodeAgent 0 should have allocated tensor"
    assert handle_1 is not None, "NodeAgent 1 should have allocated tensor"

    assert handle_0.to_dict()["tensor_size"] == [512, 768]
    assert handle_1.to_dict()["tensor_size"] == [512, 768]


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pytest.main([__file__, "-v", "--gpu"])
