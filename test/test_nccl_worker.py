"""
Tests for NCCLWorker instruction execution.

Verifies that the worker correctly executes coordinator instructions (Copy, etc.)
when given an embellished Program. Requires CUDA and the setu extensions
(_node_manager, _coordinator, _commons).
"""

import threading
import uuid

import pytest
import torch


def _get_extensions():
    """Import setu extensions; skip if not built or CUDA unavailable."""
    try:
        # Load setu package first so torch is in process (required for extension symbols)
        from setu._commons.datatypes import Device
        from setu._ir import (
            Copy,
            InitComm,
            Instruction,
            Receive,
            Send,
            ShardRef,
            generate_nccl_id,
        )
        from setu._node_manager import NCCLWorker
        from setu._planner import Participant

        return {
            "NCCLWorker": NCCLWorker,
            "Device": Device,
            "Instruction": Instruction,
            "Copy": Copy,
            "Send": Send,
            "Receive": Receive,
            "InitComm": InitComm,
            "ShardRef": ShardRef,
            "generate_nccl_id": generate_nccl_id,
            "Participant": Participant,
        }
    except ImportError as e:
        print(f"setu extensions not available: {e}")
        pytest.skip(f"setu extensions not available: {e}")


@pytest.mark.gpu
def test_nccl_worker_copy_instruction():
    """Test that NCCLWorker executes a Copy instruction (device-to-device)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ext = _get_extensions()
    NCCLWorker = ext["NCCLWorker"]
    Device = ext["Device"]
    Instruction = ext["Instruction"]
    Copy = ext["Copy"]
    ShardRef = ext["ShardRef"]

    node_id = uuid.uuid4()
    torch_device = torch.device("cuda:0")
    device = Device(torch_device)
    worker = NCCLWorker(node_id, device)
    worker.setup()

    num_elements = 128
    src = torch.randn(num_elements, device="cuda", dtype=torch.float32)
    dst = torch.zeros(num_elements, device="cuda", dtype=torch.float32)

    # ShardRef takes shard_id_str and optional tensor_name
    src_shard = ShardRef("00000000-0000-0000-0000-000000000001", "src")
    dst_shard = ShardRef("00000000-0000-0000-0000-000000000002", "dst")

    copy_instr = Copy(
        src_shard,
        0,  # src_offset_bytes
        dst_shard,
        0,  # dst_offset_bytes
        num_elements,
        torch.float32,
    )
    program = [Instruction(copy_instr)]

    # embellish receives (shard_id_string, tensor_name_optional)
    ptr_lookup = {
        "00000000-0000-0000-0000-000000000001": src.data_ptr(),
        "00000000-0000-0000-0000-000000000002": dst.data_ptr(),
    }

    for instr in program:
        instr.embellish(lambda shard_id, tensor_name: ptr_lookup[shard_id])

    worker.execute(program)

    assert torch.allclose(dst, src), "Copy instruction did not match source"


@pytest.mark.gpu
def test_nccl_worker_copy_instruction_with_offset():
    """Test Copy instruction with non-zero memory offsets (subregion copy)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ext = _get_extensions()
    NCCLWorker = ext["NCCLWorker"]
    Device = ext["Device"]
    Instruction = ext["Instruction"]
    Copy = ext["Copy"]
    ShardRef = ext["ShardRef"]

    node_id = uuid.uuid4()
    torch_device = torch.device("cuda:0")
    device = Device(torch_device)
    worker = NCCLWorker(node_id, device)
    worker.setup()

    # Buffer large enough for offset copy: copy 8 floats starting at byte 16
    n = 32
    src = torch.randn(n, device="cuda", dtype=torch.float32)
    dst = torch.zeros(n, device="cuda", dtype=torch.float32)

    # ShardRef takes shard_id_str and optional tensor_name
    src_shard = ShardRef("00000000-0000-0000-0000-000000000001", "src")
    dst_shard = ShardRef("00000000-0000-0000-0000-000000000002", "dst")

    elem_size = 4  # float32
    offset_elements = 4
    offset_bytes = offset_elements * elem_size
    num_elements = 8

    copy_instr = Copy(
        src_shard,
        offset_bytes,
        dst_shard,
        offset_bytes,
        num_elements,
        torch.float32,
    )
    program = [Instruction(copy_instr)]

    ptr_lookup = {
        "00000000-0000-0000-0000-000000000001": src.data_ptr(),
        "00000000-0000-0000-0000-000000000002": dst.data_ptr(),
    }
    for instr in program:
        instr.embellish(lambda shard_id, tensor_name: ptr_lookup[shard_id])

    worker.execute(program)

    assert torch.allclose(
        dst[offset_elements : offset_elements + num_elements],
        src[offset_elements : offset_elements + num_elements],
    ), "Subregion copy did not match"


@pytest.mark.gpu
def test_nccl_worker_empty_program():
    """Test that executing an empty program is a no-op."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ext = _get_extensions()
    NCCLWorker = ext["NCCLWorker"]
    Device = ext["Device"]

    node_id = uuid.uuid4()
    torch_device = torch.device("cuda:0")
    device = Device(torch_device)
    worker = NCCLWorker(node_id, device)
    worker.setup()

    program = []

    worker.execute(program)  # should not raise


@pytest.mark.gpu
def test_nccl_worker_send_receive():
    """Test Send/Receive between two GPUs using NCCL point-to-point."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required for send/receive test")

    ext = _get_extensions()
    NCCLWorker = ext["NCCLWorker"]
    Device = ext["Device"]
    Instruction = ext["Instruction"]
    Send = ext["Send"]
    Receive = ext["Receive"]
    InitComm = ext["InitComm"]
    ShardRef = ext["ShardRef"]
    generate_nccl_id = ext["generate_nccl_id"]
    Participant = ext["Participant"]

    # Generate node IDs for the two workers
    node_id_0 = uuid.uuid4()
    node_id_1 = uuid.uuid4()

    torch_device_0 = torch.device("cuda:0")
    device_0 = Device(torch_device_0)
    torch_device_1 = torch.device("cuda:1")
    device_1 = Device(torch_device_1)

    # Generate a shared NCCL unique ID for the communicator
    nccl_id = generate_nccl_id()

    # Participant to NCCL rank mapping (both participants)
    participant_0 = Participant(node_id_0, device_0)
    participant_1 = Participant(node_id_1, device_1)
    participant_to_rank = {participant_0: 0, participant_1: 1}

    # Create source tensor on device 0, destination on device 1
    num_elements = 128
    src = torch.randn(num_elements, device="cuda:0", dtype=torch.float32)
    dst = torch.zeros(num_elements, device="cuda:1", dtype=torch.float32)

    # ShardRef takes shard_id_str and optional tensor_name
    src_shard = ShardRef("00000000-0000-0000-0000-000000000001", "src")
    dst_shard = ShardRef("00000000-0000-0000-0000-000000000002", "dst")

    # Program for worker 0: InitComm, then Send to device 1
    init_comm_0 = InitComm(nccl_id, participant_to_rank)
    send_instr = Send(
        src_shard,
        0,  # offset
        num_elements,
        torch.float32,
        1,  # peer_rank: worker 0 sends to worker 1 (rank 1)
    )

    program_0 = [Instruction(init_comm_0), Instruction(send_instr)]

    # Program for worker 1: InitComm, then Receive from device 0
    init_comm_1 = InitComm(nccl_id, participant_to_rank)
    recv_instr = Receive(
        dst_shard,
        0,  # offset
        num_elements,
        torch.float32,
        0,  # peer_rank: worker 1 receives from worker 0 (rank 0)
    )

    program_1 = [Instruction(init_comm_1), Instruction(recv_instr)]

    # Embellish programs with device pointers
    ptr_lookup = {
        "00000000-0000-0000-0000-000000000001": src.data_ptr(),
        "00000000-0000-0000-0000-000000000002": dst.data_ptr(),
    }

    for instr in program_0:
        instr.embellish(lambda shard_id, tensor_name: ptr_lookup.get(shard_id, 0))

    for instr in program_1:
        instr.embellish(lambda shard_id, tensor_name: ptr_lookup.get(shard_id, 0))

    # Execute both programs in parallel (NCCL requires both sides to participate)
    # Workers must be created and setup on their respective threads to ensure
    # correct CUDA context ownership.
    errors = []

    def run_worker_0():
        try:
            worker = NCCLWorker(node_id_0, device_0)
            worker.setup()
            worker.execute(program_0)
        except Exception as e:
            errors.append(f"Worker 0: {e}")

    def run_worker_1():
        try:
            worker = NCCLWorker(node_id_1, device_1)
            worker.setup()
            worker.execute(program_1)
        except Exception as e:
            errors.append(f"Worker 1: {e}")

    thread_0 = threading.Thread(target=run_worker_0)
    thread_1 = threading.Thread(target=run_worker_1)

    thread_0.start()
    thread_1.start()

    thread_0.join(timeout=10)
    thread_1.join(timeout=10)

    assert not errors, f"Workers encountered errors: {errors}"
    assert not thread_0.is_alive(), "Worker 0 did not complete in time"
    assert not thread_1.is_alive(), "Worker 1 did not complete in time"

    # Verify the data was transferred correctly
    # Move dst to CPU for comparison, src to CPU as well
    assert torch.allclose(dst.cpu(), src.cpu()), "Send/Receive data mismatch"
