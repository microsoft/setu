"""
Tests for NCCLWorker instruction execution.

Verifies that the worker correctly executes coordinator instructions (Copy, etc.)
when given an embellished Program. Requires CUDA and the setu extensions
(_node_manager, _coordinator, _commons).
"""

import pytest
import torch

def _get_extensions():
    """Import setu extensions; skip if not built or CUDA unavailable."""
    try:
        # Load setu package first so torch is in process (required for extension symbols)
        from setu._commons.datatypes import Device, make_shard_id, TensorShardIdentifier
        from setu._commons.enums import DeviceKind
        from setu._coordinator.datatypes import (
            CopyInstruction,
            Instruction,
            Program,
        )
        from setu._node_manager import NCCLWorker
        return NCCLWorker, Device, DeviceKind, Program, Instruction, CopyInstruction, make_shard_id, TensorShardIdentifier
    except ImportError as e:
        print(f"setu extensions not available: {e}")
        pytest.skip(f"setu extensions not available: {e}")


@pytest.mark.gpu
def test_nccl_worker_copy_instruction():
    """Test that NCCLWorker executes a Copy instruction (device-to-device)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    (
        NCCLWorker,
        Device,
        DeviceKind,
        Program,
        Instruction,
        CopyInstruction,
        make_shard_id,
        TensorShardIdentifier,
    ) = _get_extensions()

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

    (
        NCCLWorker,
        Device,
        DeviceKind,
        Program,
        Instruction,
        CopyInstruction,
        make_shard_id,
        TensorShardIdentifier,
    ) = _get_extensions()

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
        instr.embellish(lambda n, s: ptr_lookup[(n, s)])

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

    (
        NCCLWorker,
        Device,
        DeviceKind,
        Program,
        Instruction,
        CopyInstruction,
        make_shard_id,
        TensorShardIdentifier,
    ) = _get_extensions()

    torch_device = torch.device("cuda:0")
    device = Device(DeviceKind.CUDA, 0, torch_device)
    worker = NCCLWorker(device, reply_port=0)
    worker.setup()

    program = Program()
    program.participating_workers = [0]
    program.instrs = []

    worker.execute(program)  # should not raise

