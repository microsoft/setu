"""
Tests for inproc communication between Executor and Workers.

Verifies that:
1. NCCLWorker can be created with a shared ZmqContext and inproc endpoint
2. Workers execute programs correctly when created with inproc sockets
3. Multiple workers can share the same ZmqContext with distinct inproc endpoints
4. Worker start/stop lifecycle works with inproc transport
5. NodeAgent startup creates workers with inproc transport (integration)

Requires CUDA and the setu extensions (_node_manager, _commons).
"""

import time
import uuid

import pytest
import torch


def _get_extensions():
    """Import setu extensions; skip if not built or CUDA unavailable."""
    try:
        from setu._commons.datatypes import Device
        from setu._commons.utils import ZmqContext
        from setu._ir import (
            Copy,
            Instruction,
            ShardRef,
        )
        from setu._node_manager import NCCLWorker

        return {
            "NCCLWorker": NCCLWorker,
            "Device": Device,
            "ZmqContext": ZmqContext,
            "Instruction": Instruction,
            "Copy": Copy,
            "ShardRef": ShardRef,
        }
    except ImportError as e:
        pytest.skip(f"setu extensions not available: {e}")


@pytest.mark.gpu
def test_nccl_worker_inproc_creation():
    """Test that NCCLWorker can be created with shared ZmqContext and inproc endpoint."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ext = _get_extensions()
    NCCLWorker = ext["NCCLWorker"]
    Device = ext["Device"]
    ZmqContext = ext["ZmqContext"]

    node_id = uuid.UUID(int=1)
    device = Device(torch.device("cuda:0"))
    zmq_context = ZmqContext()
    endpoint = f"inproc://test_creation_{uuid.uuid4().hex}"

    worker = NCCLWorker(node_id, device, zmq_context, endpoint)
    try:
        assert not worker.is_running(), "Worker should not be running before start"
        worker.setup()
    finally:
        del worker
        del zmq_context


@pytest.mark.gpu
def test_nccl_worker_inproc_copy_execution():
    """Test that NCCLWorker executes a Copy instruction when created with inproc."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ext = _get_extensions()
    NCCLWorker = ext["NCCLWorker"]
    Device = ext["Device"]
    ZmqContext = ext["ZmqContext"]
    Instruction = ext["Instruction"]
    Copy = ext["Copy"]
    ShardRef = ext["ShardRef"]

    node_id = uuid.UUID(int=2)
    device = Device(torch.device("cuda:0"))
    zmq_context = ZmqContext()
    endpoint = f"inproc://test_copy_{uuid.uuid4().hex}"

    worker = NCCLWorker(node_id, device, zmq_context, endpoint)
    try:
        worker.setup()

        num_elements = 128
        src = torch.randn(num_elements, device="cuda", dtype=torch.float32)
        dst = torch.zeros(num_elements, device="cuda", dtype=torch.float32)

        src_shard = ShardRef("00000000-0000-0000-0000-000000000001", "src")
        dst_shard = ShardRef("00000000-0000-0000-0000-000000000002", "dst")

        copy_instr = Copy(src_shard, 0, dst_shard, 0, num_elements, torch.float32)
        program = [Instruction(copy_instr)]

        ptr_lookup = {
            "00000000-0000-0000-0000-000000000001": src.data_ptr(),
            "00000000-0000-0000-0000-000000000002": dst.data_ptr(),
        }
        for instr in program:
            instr.embellish(lambda shard_id, _tensor_name: ptr_lookup[shard_id])

        worker.execute(program)

        assert torch.allclose(dst, src), "Copy instruction did not match source"
    finally:
        del worker
        del zmq_context


@pytest.mark.gpu
def test_nccl_worker_inproc_start_stop():
    """Test that NCCLWorker can be started and stopped with inproc transport."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ext = _get_extensions()
    NCCLWorker = ext["NCCLWorker"]
    Device = ext["Device"]
    ZmqContext = ext["ZmqContext"]

    node_id = uuid.UUID(int=3)
    device = Device(torch.device("cuda:0"))
    zmq_context = ZmqContext()
    endpoint = f"inproc://test_lifecycle_{uuid.uuid4().hex}"

    worker = NCCLWorker(node_id, device, zmq_context, endpoint)
    try:
        assert not worker.is_running()

        worker.start()
        time.sleep(0.2)
        assert worker.is_running(), "Worker should be running after start"

        worker.stop()
        assert not worker.is_running(), "Worker should not be running after stop"
    finally:
        worker.stop()
        del worker
        del zmq_context


@pytest.mark.gpu
def test_multiple_workers_shared_context():
    """Test creating multiple workers sharing the same ZmqContext."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ext = _get_extensions()
    NCCLWorker = ext["NCCLWorker"]
    Device = ext["Device"]
    ZmqContext = ext["ZmqContext"]

    zmq_context = ZmqContext()
    num_workers = min(torch.cuda.device_count(), 2)
    workers = []
    expected_devices = []

    try:
        for i in range(num_workers):
            node_id = uuid.UUID(int=10 + i)
            torch_device = torch.device(f"cuda:{i}")
            device = Device(torch_device)
            endpoint = f"inproc://test_multi_{uuid.uuid4().hex}_{i}"

            worker = NCCLWorker(node_id, device, zmq_context, endpoint)
            worker.setup()
            workers.append(worker)
            expected_devices.append(torch_device)

        for i, worker in enumerate(workers):
            assert worker.device.torch_device == expected_devices[i], (
                f"Worker {i} device mismatch: "
                f"expected {expected_devices[i]}, got {worker.device.torch_device}"
            )
    finally:
        for w in workers:
            del w
        workers.clear()
        del zmq_context


@pytest.mark.gpu
def test_worker_inproc_empty_program():
    """Test that executing an empty program is a no-op with inproc worker."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ext = _get_extensions()
    NCCLWorker = ext["NCCLWorker"]
    Device = ext["Device"]
    ZmqContext = ext["ZmqContext"]

    node_id = uuid.UUID(int=4)
    device = Device(torch.device("cuda:0"))
    zmq_context = ZmqContext()
    endpoint = f"inproc://test_empty_{uuid.uuid4().hex}"

    worker = NCCLWorker(node_id, device, zmq_context, endpoint)
    try:
        worker.setup()
        worker.execute([])  # should not raise
    finally:
        del worker
        del zmq_context


@pytest.mark.gpu
def test_node_agent_creates_workers_with_inproc():
    """
    Integration test: verify NodeAgent startup succeeds with the new
    Executor-owned workers using inproc transport.

    This test starts a Coordinator + NodeAgent and verifies the NodeAgent
    starts and stops cleanly (which exercises the full
    Executor::CreateWorkers() -> inproc bind/connect path).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import torch.multiprocessing as mp

    def _run_coordinator(port, ready_event, stop_event):
        from setu._coordinator import Coordinator

        coordinator = Coordinator(port)
        coordinator.start()
        ready_event.set()
        while not stop_event.is_set():
            time.sleep(0.05)
        coordinator.stop()

    def _run_node_agent(port, coordinator_endpoint, ready_event, stop_event):
        from setu._commons.datatypes import Device
        from setu._node_manager import NodeAgent

        node_id = uuid.UUID(int=100)
        devices = [Device(torch_device=torch.device("cuda:0"))]
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

    coordinator_port = 29500
    node_agent_port = 29501
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()

    coordinator_proc = ctx.Process(
        target=_run_coordinator,
        args=(coordinator_port, coordinator_ready, stop_event),
    )
    node_agent_proc = ctx.Process(
        target=_run_node_agent,
        args=(node_agent_port, coordinator_endpoint, node_agent_ready, stop_event),
    )

    try:
        coordinator_proc.start()
        assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

        node_agent_proc.start()
        assert node_agent_ready.wait(timeout=10), "NodeAgent failed to start"

        # NodeAgent started successfully -- this means Executor::CreateWorkers()
        # bound inproc sockets and workers started without errors.
        time.sleep(0.5)
    finally:
        stop_event.set()
        time.sleep(0.1)
        for proc in [node_agent_proc, coordinator_proc]:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

    assert not node_agent_proc.is_alive(), "NodeAgent process did not exit cleanly"
    assert not coordinator_proc.is_alive(), "Coordinator process did not exit cleanly"
