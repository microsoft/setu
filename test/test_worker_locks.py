"""
Integration tests for tensor shard read/write locks during worker execution.

Tests that the executor correctly acquires and releases file-based read/write
locks when executing programs. Verifies:
- Copy operations complete correctly with locking (end-to-end)
- A client write lock on a destination shard delays copy execution
- A client read lock on a source shard coexists with executor read locks
- Locks are released after execution, allowing client access
"""

import time
import uuid

import pytest
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor


def _run_coordinator(port: int, ready_event, stop_event):
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
    device_index: int = 0,
):
    """Run the NodeAgent in a separate process."""
    from setu._commons.datatypes import Device
    from setu._node_manager import NodeAgent

    if node_id is None:
        node_id = uuid.uuid4()

    devices = [Device(torch_device=torch.device(f"cuda:{device_index}"))]

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
    return rebuild_cuda_tensor(
        **spec_dict,
        tensor_cls=torch.Tensor,
        storage_cls=torch.storage.UntypedStorage,
    )


@pytest.fixture(scope="module")
def infrastructure():
    """Start Coordinator and NodeAgent once for all tests in this module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_port = 29700
    node_agent_port = 29800
    coordinator_endpoint = f"tcp://localhost:{coordinator_port}"
    client_endpoint = f"tcp://localhost:{node_agent_port}"

    ctx = mp.get_context("spawn")
    coordinator_ready = ctx.Event()
    node_agent_ready = ctx.Event()
    stop_event = ctx.Event()

    coordinator_proc = ctx.Process(
        target=_run_coordinator,
        args=(coordinator_port, coordinator_ready, stop_event),
    )
    coordinator_proc.start()
    assert coordinator_ready.wait(timeout=10), "Coordinator failed to start"

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

    time.sleep(0.1)

    yield {"client_endpoint": client_endpoint}

    stop_event.set()
    time.sleep(0.1)
    for proc in [node_agent_proc, coordinator_proc]:
        proc.join(timeout=3)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)


# ==============================================================================
# Helper: register + write + pull in subprocess
# ==============================================================================


def _source_register_and_write(
    client_endpoint: str,
    tensor_name: str,
    dims_data: list,
    fill_value: float,
    ready_event,
    result_queue,
    device_index: int = 0,
):
    """Register source shard, write data, then stay alive."""
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    try:
        client = Client()
        client.connect(client_endpoint)

        dims_spec = [
            TensorDimSpec(name, size, start, end)
            for name, size, start, end in dims_data
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
        tensor = _rebuild_tensor_from_handle(tensor_ipc_spec)
        tensor.fill_(fill_value)
        torch.cuda.synchronize()

        result_queue.put({"success": True, "shard_id": shard_ref.shard_id})
        ready_event.set()

        # Stay alive
        while True:
            time.sleep(0.1)

    except Exception as e:
        import traceback

        result_queue.put(
            {"success": False, "error": str(e), "tb": traceback.format_exc()}
        )
        ready_event.set()


def _dest_pull_and_verify(
    client_endpoint: str,
    src_tensor_name: str,
    dst_tensor_name: str,
    dims_data: list,
    expected_value: float,
    source_ready_event,
    result_queue,
    device_index: int = 0,
):
    """Wait for source, register dest shard, pull data, verify value."""
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
        if not source_ready_event.wait(timeout=30):
            raise RuntimeError("Timeout waiting for source")

        client = Client()
        client.connect(client_endpoint)

        dims_spec = [
            TensorDimSpec(name, size, start, end)
            for name, size, start, end in dims_data
        ]
        device = Device(torch_device=torch.device(f"cuda:{device_index}"))
        shard_spec = TensorShardSpec(
            name=dst_tensor_name, dims=dims_spec, dtype=torch.float32, device=device
        )

        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            raise RuntimeError(f"Failed to register dest shard {dst_tensor_name}")
        client.wait_for_shard_allocation(shard_ref.shard_id)

        # Build CopySpec for pull
        dim_map = {name: TensorDim(name, size) for name, size, _, _ in dims_data}
        src_sel = TensorSelection(src_tensor_name, dim_map)
        dst_sel = TensorSelection(dst_tensor_name, dim_map)
        copy_spec = CopySpec(src_tensor_name, dst_tensor_name, src_sel, dst_sel)

        copy_op_id = client.submit_pull(copy_spec)
        if copy_op_id is None:
            raise RuntimeError("Failed to submit pull")

        client.wait_for_copy(copy_op_id)

        # Verify
        tensor_ipc_spec, _, _ = client.get_tensor_handle(shard_ref)
        tensor = _rebuild_tensor_from_handle(tensor_ipc_spec)
        torch.cuda.synchronize()
        actual = tensor.mean().item()
        matches = abs(actual - expected_value) < 1e-5

        result_queue.put({"success": True, "matches": matches, "actual": actual})
        client.disconnect()

    except Exception as e:
        import traceback

        result_queue.put(
            {"success": False, "error": str(e), "tb": traceback.format_exc()}
        )


# ==============================================================================
# Tests
# ==============================================================================


@pytest.mark.gpu
def test_pull_correctness_with_locks(infrastructure):
    """
    End-to-end test: pull operation transfers data correctly with shard locking.

    The executor acquires read lock on source, write lock on destination,
    dispatches to the worker, and releases both after completion.
    """
    client_endpoint = infrastructure["client_endpoint"]
    dims_data = [("dim_0", 8, 0, 8), ("dim_1", 16, 0, 16)]
    fill_value = 42.0

    ctx = mp.get_context("spawn")
    source_ready = ctx.Event()
    src_queue = ctx.Queue()
    dst_queue = ctx.Queue()

    src_proc = ctx.Process(
        target=_source_register_and_write,
        args=(
            client_endpoint,
            "lock_src_1",
            dims_data,
            fill_value,
            source_ready,
            src_queue,
        ),
    )
    src_proc.start()

    dst_proc = ctx.Process(
        target=_dest_pull_and_verify,
        args=(
            client_endpoint,
            "lock_src_1",
            "lock_dst_1",
            dims_data,
            fill_value,
            source_ready,
            dst_queue,
        ),
    )
    dst_proc.start()

    dst_proc.join(timeout=30)
    src_proc.terminate()
    src_proc.join(timeout=3)

    src_result = src_queue.get()
    assert src_result["success"], f"Source failed: {src_result.get('error')}"

    dst_result = dst_queue.get()
    assert dst_result["success"], f"Dest failed: {dst_result.get('error')}"
    assert dst_result[
        "matches"
    ], f"Pull data mismatch: expected {fill_value}, got {dst_result['actual']}"


@pytest.mark.gpu
def test_read_lock_released_after_pull(infrastructure):
    """
    After a pull completes, the executor releases locks and a client
    can freely write to the destination tensor.
    """
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]
    dims_data = [("dim_0", 4, 0, 4), ("dim_1", 8, 0, 8)]
    fill_value = 10.0

    ctx = mp.get_context("spawn")
    source_ready = ctx.Event()
    src_queue = ctx.Queue()
    dst_queue = ctx.Queue()

    # Source registers and writes fill_value
    src_proc = ctx.Process(
        target=_source_register_and_write,
        args=(
            client_endpoint,
            "lock_release_src",
            dims_data,
            fill_value,
            source_ready,
            src_queue,
        ),
    )
    src_proc.start()

    # Dest registers and pulls
    dst_proc = ctx.Process(
        target=_dest_pull_and_verify,
        args=(
            client_endpoint,
            "lock_release_src",
            "lock_release_dst",
            dims_data,
            fill_value,
            source_ready,
            dst_queue,
        ),
    )
    dst_proc.start()

    dst_proc.join(timeout=30)
    src_proc.terminate()
    src_proc.join(timeout=3)

    dst_result = dst_queue.get()
    assert dst_result["success"], f"Pull failed: {dst_result.get('error')}"
    assert dst_result["matches"], "Pull data should match"

    # After pull completes, locks are released.
    # A new client should be able to write to the destination tensor.
    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec(name, size, start, end) for name, size, start, end in dims_data
    ]
    shard_spec = TensorShardSpec(
        name="lock_release_dst", dims=dims, dtype=torch.float32, device=device
    )
    shard_ref = client.register_tensor_shard(shard_spec)

    # If the executor still held locks, this would deadlock
    if shard_ref is not None:
        with client.write(shard_ref) as tensor:
            tensor.fill_(99.0)

        with client.read(shard_ref) as tensor:
            assert torch.allclose(
                tensor, torch.full_like(tensor, 99.0)
            ), "Client should be able to write after executor releases locks"

    client.disconnect()


def _register_lock_pull_release(
    client_endpoint: str,
    src_tensor_name: str,
    dst_tensor_name: str,
    dims_data: list,
    hold_seconds: float,
    source_ready_event,
    lock_acquired_event,
    result_queue,
    device_index: int = 0,
):
    """Register dest shard, acquire write lock, submit pull, hold lock, release, wait.

    The pull is submitted while the write lock is held. Since submit_pull is
    non-blocking (it just sends a request), the executor will start trying to
    acquire the write lock and block. After hold_seconds we release the lock,
    letting the executor complete.
    """
    from setu._client import Client
    from setu._commons.datatypes import (
        CopySpec,
        Device,
        TensorDim,
        TensorDimSpec,
        TensorSelection,
        TensorShard,
        TensorShardSpec,
    )
    from setu._commons.datatypes import TensorShardWriteHandle as NativeWriteHandle

    try:
        if not source_ready_event.wait(timeout=30):
            raise RuntimeError("Timeout waiting for source")

        client = Client()
        client.connect(client_endpoint)

        device = Device(torch_device=torch.device(f"cuda:{device_index}"))
        dims = [
            TensorDimSpec(name, size, start, end)
            for name, size, start, end in dims_data
        ]
        shard_spec = TensorShardSpec(
            name=dst_tensor_name, dims=dims, dtype=torch.float32, device=device
        )
        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is None:
            raise RuntimeError(f"Failed to register shard {dst_tensor_name}")

        client.wait_for_shard_allocation(shard_ref.shard_id)

        # Build TensorShard for file lock
        tensor_ipc_spec, metadata, lock_base_dir = client.get_tensor_handle(shard_ref)
        tensor = _rebuild_tensor_from_handle(tensor_ipc_spec)
        shard = TensorShard(
            metadata=metadata, tensor=tensor, lock_base_dir=lock_base_dir
        )

        # Acquire exclusive write lock
        write_handle = NativeWriteHandle(shard)
        lock_acquired_event.set()

        # Submit pull while holding the lock (non-blocking — just sends request)
        dim_map = {name: TensorDim(name, size) for name, size, _, _ in dims_data}
        src_sel = TensorSelection(src_tensor_name, dim_map)
        dst_sel = TensorSelection(dst_tensor_name, dim_map)
        copy_spec = CopySpec(src_tensor_name, dst_tensor_name, src_sel, dst_sel)

        start_time = time.time()
        copy_op_id = client.submit_pull(copy_spec)
        if copy_op_id is None:
            raise RuntimeError("Failed to submit pull")

        # Hold lock — executor is blocked waiting for it
        time.sleep(hold_seconds)

        # Release lock — executor can now proceed
        del write_handle

        # Wait for copy to complete (should finish quickly after lock release)
        client.wait_for_copy(copy_op_id)
        elapsed = time.time() - start_time

        result_queue.put({"success": True, "elapsed": elapsed})
        client.disconnect()

    except Exception as e:
        import traceback

        lock_acquired_event.set()
        result_queue.put(
            {"success": False, "error": str(e), "tb": traceback.format_exc()}
        )


@pytest.mark.gpu
def test_client_write_lock_blocks_executor_write(infrastructure):
    """
    A client holding a write lock on a destination shard should block the
    executor from acquiring its write lock for a pull into that shard.

    Scenario:
    1. Source client registers + writes source tensor
    2. Same process: registers dest, acquires write lock, submits pull
    3. Pull is submitted (non-blocking) while lock is held
    4. After hold_seconds, lock is released — executor can proceed
    5. wait_for_copy returns

    We verify the total elapsed time from submit to completion >= hold_seconds.
    """
    client_endpoint = infrastructure["client_endpoint"]
    dims_data = [("dim_0", 4, 0, 4), ("dim_1", 4, 0, 4)]
    fill_value = 77.0
    hold_seconds = 2.0

    ctx = mp.get_context("spawn")
    source_ready = ctx.Event()
    lock_acquired = ctx.Event()
    src_queue = ctx.Queue()
    pull_queue = ctx.Queue()

    # Step 1: Source registers and writes
    src_proc = ctx.Process(
        target=_source_register_and_write,
        args=(
            client_endpoint,
            "block_src",
            dims_data,
            fill_value,
            source_ready,
            src_queue,
        ),
    )
    src_proc.start()

    # Step 2: Single process registers dest, holds write lock, submits pull,
    # releases lock after hold_seconds, waits for completion.
    # SubmitPull requires the same Client that registered the dest shard.
    pull_proc = ctx.Process(
        target=_register_lock_pull_release,
        args=(
            client_endpoint,
            "block_src",
            "block_dst",
            dims_data,
            hold_seconds,
            source_ready,
            lock_acquired,
            pull_queue,
        ),
    )
    pull_proc.start()

    pull_proc.join(timeout=30)
    src_proc.terminate()
    src_proc.join(timeout=3)

    src_result = src_queue.get()
    assert src_result["success"], f"Source failed: {src_result.get('error')}"

    pull_result = pull_queue.get()
    assert pull_result["success"], f"Pull failed: {pull_result.get('error')}"

    # The pull should have been delayed by the write lock
    elapsed = pull_result["elapsed"]
    assert elapsed >= hold_seconds * 0.8, (
        f"Pull completed in {elapsed:.2f}s, expected >= {hold_seconds * 0.8:.2f}s "
        f"(lock was held for {hold_seconds}s)"
    )


@pytest.mark.gpu
def test_concurrent_pulls_same_source(infrastructure):
    """
    Multiple concurrent pulls from the same source should succeed.

    The executor acquires read locks on the source shard. Since read locks
    are shared (sharable), multiple concurrent executions reading the same
    source should not block each other.
    """
    client_endpoint = infrastructure["client_endpoint"]
    dims_data = [("dim_0", 4, 0, 4), ("dim_1", 8, 0, 8)]
    fill_value = 55.0
    num_dest_clients = 3

    ctx = mp.get_context("spawn")
    source_ready = ctx.Event()
    src_queue = ctx.Queue()

    # Source registers and writes
    src_proc = ctx.Process(
        target=_source_register_and_write,
        args=(
            client_endpoint,
            "concurrent_src",
            dims_data,
            fill_value,
            source_ready,
            src_queue,
        ),
    )
    src_proc.start()
    assert source_ready.wait(timeout=15), "Source failed to start"

    # Launch multiple destination clients concurrently
    dst_procs = []
    dst_queues = []
    for i in range(num_dest_clients):
        q = ctx.Queue()
        dst_queues.append(q)
        proc = ctx.Process(
            target=_dest_pull_and_verify,
            args=(
                client_endpoint,
                "concurrent_src",
                f"concurrent_dst_{i}",
                dims_data,
                fill_value,
                source_ready,
                q,
            ),
        )
        proc.start()
        dst_procs.append(proc)

    # Wait for all to complete
    for proc in dst_procs:
        proc.join(timeout=30)

    src_proc.terminate()
    src_proc.join(timeout=3)

    # Verify all pulls succeeded with correct data
    for i, q in enumerate(dst_queues):
        result = q.get()
        assert result["success"], f"Dest {i} failed: {result.get('error')}"
        assert result[
            "matches"
        ], f"Dest {i} data mismatch: expected {fill_value}, got {result['actual']}"


@pytest.mark.gpu
def test_sequential_pulls_lock_cycle(infrastructure):
    """
    Sequential pull operations should correctly acquire and release locks
    each time. After each pull, the destination should be freely accessible.
    """
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]
    dims_data = [("dim_0", 4, 0, 4), ("dim_1", 4, 0, 4)]

    ctx = mp.get_context("spawn")

    for iteration, fill_value in enumerate([10.0, 20.0, 30.0]):
        src_name = f"seq_src_{iteration}"
        dst_name = f"seq_dst_{iteration}"

        source_ready = ctx.Event()
        src_queue = ctx.Queue()
        dst_queue = ctx.Queue()

        src_proc = ctx.Process(
            target=_source_register_and_write,
            args=(
                client_endpoint,
                src_name,
                dims_data,
                fill_value,
                source_ready,
                src_queue,
            ),
        )
        src_proc.start()

        dst_proc = ctx.Process(
            target=_dest_pull_and_verify,
            args=(
                client_endpoint,
                src_name,
                dst_name,
                dims_data,
                fill_value,
                source_ready,
                dst_queue,
            ),
        )
        dst_proc.start()

        dst_proc.join(timeout=30)
        src_proc.terminate()
        src_proc.join(timeout=3)

        dst_result = dst_queue.get()
        assert dst_result[
            "success"
        ], f"Iteration {iteration} pull failed: {dst_result.get('error')}"
        assert dst_result["matches"], (
            f"Iteration {iteration} data mismatch: "
            f"expected {fill_value}, got {dst_result['actual']}"
        )

        # After pull, verify we can freely read/write via a new client
        client = Client(client_endpoint)
        device = Device(torch_device=torch.device("cuda:0"))
        dims = [TensorDimSpec(n, s, st, e) for n, s, st, e in dims_data]
        shard_spec = TensorShardSpec(
            name=dst_name, dims=dims, dtype=torch.float32, device=device
        )
        shard_ref = client.register_tensor_shard(shard_spec)
        if shard_ref is not None:
            with client.read(shard_ref) as tensor:
                assert (
                    tensor is not None
                ), f"Iteration {iteration}: should read after executor lock release"
        client.disconnect()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pytest.main([__file__, "-v", "--gpu"])
