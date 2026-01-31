"""
Tests for NCCLWorker instruction execution.

Verifies that the worker correctly executes coordinator instructions (Copy, etc.)
when given an embellished Program. Requires CUDA and the setu extensions
(_node_manager, _coordinator, _commons).
"""

import threading

import pytest
import torch

def _get_extensions():
    """Import setu extensions; skip if not built or CUDA unavailable."""
    try:
        # Load setu package first so torch is in process (required for extension symbols)
        from setu._commons.datatypes import Device, make_shard_id, TensorShardIdentifier
        from setu._commons.enums import DeviceKind
        from setu._coordinator.datatypes import Program
        from setu._ir import (
            CopyInstruction,
            InitCommInstruction,
            Instruction,
            ReceiveInstruction,
            SendInstruction,
            generate_nccl_id,
        )
        from setu._node_manager import NCCLWorker
        return {
            "NCCLWorker": NCCLWorker,
            "Device": Device,
            "DeviceKind": DeviceKind,
            "Program": Program,
            "Instruction": Instruction,
            "CopyInstruction": CopyInstruction,
            "SendInstruction": SendInstruction,
            "ReceiveInstruction": ReceiveInstruction,
            "InitCommInstruction": InitCommInstruction,
            "make_shard_id": make_shard_id,
            "TensorShardIdentifier": TensorShardIdentifier,
            "generate_nccl_id": generate_nccl_id,
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
    DeviceKind = ext["DeviceKind"]
    Program = ext["Program"]
    Instruction = ext["Instruction"]
    CopyInstruction = ext["CopyInstruction"]
    make_shard_id = ext["make_shard_id"]
    TensorShardIdentifier = ext["TensorShardIdentifier"]

    torch_device = torch.device("cuda:0")
    device = Device(DeviceKind.CUDA, 0, torch_device)
    worker = NCCLWorker(device, reply_port=0)
    worker.setup()

    num_elements = 128
    src = torch.randn(num_elements, device="cuda", dtype=torch.float32)
    dst = torch.zeros(num_elements, device="cuda", dtype=torch.float32)

    shard_id_src = make_shard_id("00000000-0000-0000-0000-000000000001")
    shard_id_dst = make_shard_id("00000000-0000-0000-0000-000000000002")

    copy_instr = CopyInstruction(
        TensorShardIdentifier("src", shard_id_src),
        0,
        TensorShardIdentifier("dst", shard_id_dst),
        0,
        torch.float32,
        num_elements,
    )
    program = Program()
    program.participating_workers = [0]
    program.instrs = [Instruction(copy_instr)]

    ptr_lookup = {
        ("src", "00000000-0000-0000-0000-000000000001"): src.data_ptr(),
        ("dst", "00000000-0000-0000-0000-000000000002"): dst.data_ptr(),
    }

    for instr in program.instrs:
        instr.embellish(lambda n, s: ptr_lookup[(n, s)])

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
    DeviceKind = ext["DeviceKind"]
    Program = ext["Program"]
    Instruction = ext["Instruction"]
    CopyInstruction = ext["CopyInstruction"]
    make_shard_id = ext["make_shard_id"]
    TensorShardIdentifier = ext["TensorShardIdentifier"]

    torch_device = torch.device("cuda:0")
    device = Device(DeviceKind.CUDA, 0, torch_device)
    worker = NCCLWorker(device, reply_port=0)
    worker.setup()

    # Buffer large enough for offset copy: copy 8 floats starting at byte 16
    n = 32
    src = torch.randn(n, device="cuda", dtype=torch.float32)
    dst = torch.zeros(n, device="cuda", dtype=torch.float32)

    shard_id_src = make_shard_id("00000000-0000-0000-0000-000000000001")
    shard_id_dst = make_shard_id("00000000-0000-0000-0000-000000000002")

    elem_size = 4  # float32
    offset_elements = 4
    offset_bytes = offset_elements * elem_size
    num_elements = 8

    copy_instr = CopyInstruction(
        TensorShardIdentifier("src", shard_id_src),
        offset_bytes,
        TensorShardIdentifier("dst", shard_id_dst),
        offset_bytes,
        torch.float32,
        num_elements,
    )
    program = Program()
    program.participating_workers = [0]
    program.instrs = [Instruction(copy_instr)]

    ptr_lookup = {
        ("src", "00000000-0000-0000-0000-000000000001"): src.data_ptr(),
        ("dst", "00000000-0000-0000-0000-000000000002"): dst.data_ptr(),
    }
    for instr in program.instrs:
        instr.embellish(lambda name, shard: ptr_lookup[(name, shard)])

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
    DeviceKind = ext["DeviceKind"]
    Program = ext["Program"]

    torch_device = torch.device("cuda:0")
    device = Device(DeviceKind.CUDA, 0, torch_device)
    worker = NCCLWorker(device, reply_port=0)
    worker.setup()

    program = Program()
    program.participating_workers = [0]
    program.instrs = []

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
    DeviceKind = ext["DeviceKind"]
    Program = ext["Program"]
    Instruction = ext["Instruction"]
    SendInstruction = ext["SendInstruction"]
    ReceiveInstruction = ext["ReceiveInstruction"]
    InitCommInstruction = ext["InitCommInstruction"]
    make_shard_id = ext["make_shard_id"]
    TensorShardIdentifier = ext["TensorShardIdentifier"]
    generate_nccl_id = ext["generate_nccl_id"]

    # print("\n=== Starting send/receive test ===")

    # Create workers for device 0 and device 1
    torch_device_0 = torch.device("cuda:0")
    torch_device_1 = torch.device("cuda:1")
    # print(f"Creating devices: cuda:0 and cuda:1")

    device_0 = Device(DeviceKind.CUDA, 0, torch_device_0)
    device_1 = Device(DeviceKind.CUDA, 1, torch_device_1)

    # Generate a shared NCCL unique ID for the communicator
    nccl_id = generate_nccl_id()
    # print(f"Generated NCCL ID: {nccl_id}")

    # Device rank to NCCL rank mapping (both devices participate)
    device_to_rank = {0: 0, 1: 1}
    # print(f"Device to rank mapping: {device_to_rank}")

    # Create source tensor on device 0, destination on device 1
    num_elements = 128
    src = torch.randn(num_elements, device="cuda:0", dtype=torch.float32)
    dst = torch.zeros(num_elements, device="cuda:1", dtype=torch.float32)
    # print(f"Created tensors: src on cuda:0, dst on cuda:1, {num_elements} elements")
    # print(f"src[:5] = {src[:5]}")
    # print(f"dst[:5] = {dst[:5]}")

    shard_id_src = make_shard_id("00000000-0000-0000-0000-000000000001")
    shard_id_dst = make_shard_id("00000000-0000-0000-0000-000000000002")

    # Program for worker 0: InitComm, then Send to device 1
    init_comm_0 = InitCommInstruction(nccl_id, device_to_rank)
    send_instr = SendInstruction(
        1,  # dst_device_id
        TensorShardIdentifier("src", shard_id_src),
        torch.float32,
        0,  # memory_offset_bytes
        num_elements,
    )
    # print(f"Created SendInstruction: {send_instr}")

    program_0 = Program()
    program_0.participating_workers = [0, 1]
    program_0.instrs = [Instruction(init_comm_0), Instruction(send_instr)]
    # print(f"Program 0 has {len(program_0.instrs)} instructions")

    # Program for worker 1: InitComm, then Receive from device 0
    init_comm_1 = InitCommInstruction(nccl_id, device_to_rank)
    recv_instr = ReceiveInstruction(
        0,  # src_device_id
        TensorShardIdentifier("dst", shard_id_dst),
        torch.float32,
        0,  # memory_offset_bytes
        num_elements,
    )
    # print(f"Created ReceiveInstruction: {recv_instr}")

    program_1 = Program()
    program_1.participating_workers = [0, 1]
    program_1.instrs = [Instruction(init_comm_1), Instruction(recv_instr)]
    # print(f"Program 1 has {len(program_1.instrs)} instructions")

    # Embellish programs with device pointers
    ptr_lookup_0 = {
        ("src", "00000000-0000-0000-0000-000000000001"): src.data_ptr(),
    }
    ptr_lookup_1 = {
        ("dst", "00000000-0000-0000-0000-000000000002"): dst.data_ptr(),
    }
    # print(f"ptr_lookup_0: {ptr_lookup_0}")
    # print(f"ptr_lookup_1: {ptr_lookup_1}")

    # print("Embellishing program 0...")
    for instr in program_0.instrs:
        instr.embellish(lambda name, shard: ptr_lookup_0.get((name, shard), 0))

    # print("Embellishing program 1...")
    for instr in program_1.instrs:
        instr.embellish(lambda name, shard: ptr_lookup_1.get((name, shard), 0))

    # Execute both programs in parallel (NCCL requires both sides to participate)
    # Workers must be created and setup on their respective threads to ensure
    # correct CUDA context ownership.
    errors = []
    workers = {}

    def run_worker_0():
        try:
            # print("[Thread 0] Creating and setting up worker...")
            worker = NCCLWorker(device_0, reply_port=0)
            worker.setup()
            workers[0] = worker
            # print("[Thread 0] Starting execution...")
            worker.execute(program_0)
            # print("[Thread 0] Execution complete")
        except Exception as e:
            errors.append(f"Worker 0: {e}")

    def run_worker_1():
        try:
            # print("[Thread 1] Creating and setting up worker...")
            worker = NCCLWorker(device_1, reply_port=0)
            worker.setup()
            workers[1] = worker
            # print("[Thread 1] Starting execution...")
            worker.execute(program_1)
            # print("[Thread 1] Execution complete")
        except Exception as e:
            errors.append(f"Worker 1: {e}")

    # print("Starting threads...")
    thread_0 = threading.Thread(target=run_worker_0)
    thread_1 = threading.Thread(target=run_worker_1)

    thread_0.start()
    thread_1.start()

    # print("Waiting for threads to complete (timeout=10s)...")
    thread_0.join(timeout=10)
    thread_1.join(timeout=10)

    # print(f"Thread 0 alive: {thread_0.is_alive()}")
    # print(f"Thread 1 alive: {thread_1.is_alive()}")
    # print(f"Errors: {errors}")

    assert not errors, f"Workers encountered errors: {errors}"
    assert not thread_0.is_alive(), "Worker 0 did not complete in time"
    assert not thread_1.is_alive(), "Worker 1 did not complete in time"

    # Verify the data was transferred correctly
    # print(f"After transfer - dst[:5] = {dst[:5]}")
    # print(f"Expected (src[:5]) = {src[:5]}")

    # Move dst to CPU for comparison, src to CPU as well
    assert torch.allclose(dst.cpu(), src.cpu()), "Send/Receive data mismatch"
    # print("=== Test passed! ===")
