"""
Unit tests for Python Client tensor handles API.

Tests the TensorReadHandle and TensorWriteHandle context managers
for thread-safe tensor shard access.
"""

import time
import uuid
from test.fixtures import ClusterSpec, DeviceSpec, SetuTestCluster

import pytest
import torch
import torch.multiprocessing as mp

from setu._commons.datatypes import Device


@pytest.fixture(scope="module")
def infrastructure():
    """
    Start Coordinator and NodeAgent once for all tests in this module.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    node_id = uuid.uuid4()
    spec = ClusterSpec(
        coordinator_port=29300,
        nodes={node_id: (29301, [DeviceSpec(Device(torch.device("cuda:0")))])},
    )

    with SetuTestCluster(spec) as cluster:
        yield {
            "client_endpoint": cluster.client_endpoint(node_id),
        }


@pytest.mark.gpu
def test_write_and_read_tensor(infrastructure):
    """Test writing to a tensor shard and reading it back."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    # Create client using Python API
    client = Client(client_endpoint)

    # Register a tensor shard
    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 4, 0, 4),
        TensorDimSpec("dim_1", 8, 0, 8),
    ]
    shard_spec = TensorShardSpec(
        name="test_rw_tensor",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None, "Should receive a valid TensorShardRef"

    # Write data to tensor using write context manager
    test_value = 42.0
    with client.write(shard_ref) as tensor:
        assert tensor.shape == (4, 8), f"Unexpected shape: {tensor.shape}"
        tensor.fill_(test_value)

    # Read data back using read context manager
    with client.read(shard_ref) as tensor:
        assert tensor.shape == (4, 8), f"Unexpected shape: {tensor.shape}"
        assert torch.allclose(
            tensor, torch.full((4, 8), test_value, device="cuda:0")
        ), "Tensor values should match what was written"

    client.disconnect()


@pytest.mark.gpu
def test_write_specific_values(infrastructure):
    """Test writing specific values to different positions."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 8, 0, 8),
        TensorDimSpec("cols", 16, 0, 16),
    ]
    shard_spec = TensorShardSpec(
        name="test_specific_values",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    # Write specific pattern
    with client.write(shard_ref) as tensor:
        tensor.zero_()
        tensor[0, :] = 1.0
        tensor[1, :] = 2.0
        tensor[:, 0] = 10.0

    # Verify the pattern
    with client.read(shard_ref) as tensor:
        # First column should be 10.0 everywhere
        assert torch.allclose(
            tensor[:, 0], torch.full((8,), 10.0, device="cuda:0")
        ), "First column should be 10.0"

        # Row 0, col > 0 should be 1.0
        assert torch.allclose(
            tensor[0, 1:], torch.full((15,), 1.0, device="cuda:0")
        ), "Row 0 (except col 0) should be 1.0"

        # Row 1, col > 0 should be 2.0
        assert torch.allclose(
            tensor[1, 1:], torch.full((15,), 2.0, device="cuda:0")
        ), "Row 1 (except col 0) should be 2.0"

    client.disconnect()


@pytest.mark.gpu
def test_multiple_writes_same_tensor(infrastructure):
    """Test multiple sequential writes to the same tensor."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 4, 0, 4),
        TensorDimSpec("dim_1", 4, 0, 4),
    ]
    shard_spec = TensorShardSpec(
        name="test_multi_write",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    # First write
    with client.write(shard_ref) as tensor:
        tensor.fill_(1.0)

    with client.read(shard_ref) as tensor:
        assert tensor.sum().item() == 16.0, "Should be 16.0 after first write"

    # Second write - overwrite with new value
    with client.write(shard_ref) as tensor:
        tensor.fill_(3.0)

    with client.read(shard_ref) as tensor:
        assert tensor.sum().item() == 48.0, "Should be 48.0 after second write"

    # Third write - increment
    with client.write(shard_ref) as tensor:
        tensor.add_(1.0)

    with client.read(shard_ref) as tensor:
        assert tensor.sum().item() == 64.0, "Should be 64.0 after increment"

    client.disconnect()


@pytest.mark.gpu
def test_tensor_dtype_float16(infrastructure):
    """Test tensor handles with float16 dtype."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 8, 0, 8),
        TensorDimSpec("dim_1", 8, 0, 8),
    ]
    shard_spec = TensorShardSpec(
        name="test_float16_tensor",
        dims=dims,
        dtype=torch.float16,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    with client.write(shard_ref) as tensor:
        assert tensor.dtype == torch.float16, f"Expected float16, got {tensor.dtype}"
        tensor.fill_(0.5)

    with client.read(shard_ref) as tensor:
        assert tensor.dtype == torch.float16
        # float16 has limited precision, use appropriate tolerance
        expected = torch.full((8, 8), 0.5, dtype=torch.float16, device="cuda:0")
        assert torch.allclose(tensor, expected, rtol=1e-3), "Values should match"

    client.disconnect()


@pytest.mark.gpu
def test_tensor_dtype_bfloat16(infrastructure):
    """Test tensor handles with bfloat16 dtype."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 8, 0, 8),
        TensorDimSpec("dim_1", 8, 0, 8),
    ]
    shard_spec = TensorShardSpec(
        name="test_bfloat16_tensor",
        dims=dims,
        dtype=torch.bfloat16,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    with client.write(shard_ref) as tensor:
        assert tensor.dtype == torch.bfloat16, f"Expected bfloat16, got {tensor.dtype}"
        tensor.fill_(1.5)

    with client.read(shard_ref) as tensor:
        assert tensor.dtype == torch.bfloat16
        expected = torch.full((8, 8), 1.5, dtype=torch.bfloat16, device="cuda:0")
        assert torch.allclose(tensor, expected, rtol=1e-2), "Values should match"

    client.disconnect()


@pytest.mark.gpu
def test_large_tensor(infrastructure):
    """Test tensor handles with a larger tensor."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("batch", 32, 0, 32),
        TensorDimSpec("seq", 128, 0, 128),
        TensorDimSpec("hidden", 256, 0, 256),
    ]
    shard_spec = TensorShardSpec(
        name="test_large_tensor",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    # Write random data
    random_data = torch.randn(32, 128, 256, device="cuda:0")
    with client.write(shard_ref) as tensor:
        assert tensor.shape == (32, 128, 256), f"Unexpected shape: {tensor.shape}"
        tensor.copy_(random_data)

    # Read and verify
    with client.read(shard_ref) as tensor:
        assert torch.allclose(tensor, random_data), "Large tensor data should match"

    client.disconnect()


@pytest.mark.gpu
def test_read_without_prior_write(infrastructure):
    """Test reading a tensor that hasn't been explicitly written (should have zeros or uninitialized)."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 4, 0, 4),
        TensorDimSpec("dim_1", 4, 0, 4),
    ]
    shard_spec = TensorShardSpec(
        name="test_uninitialized_tensor",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    # Read should work even without prior write
    with client.read(shard_ref) as tensor:
        assert tensor.shape == (4, 4), f"Unexpected shape: {tensor.shape}"
        # Tensor may have uninitialized values, just check we can read it

    client.disconnect()


@pytest.mark.gpu
def test_nested_read_same_tensor(infrastructure):
    """Test that nested reads on the same tensor work (shared locks)."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 4, 0, 4),
        TensorDimSpec("dim_1", 4, 0, 4),
    ]
    shard_spec = TensorShardSpec(
        name="test_nested_read",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    # Write initial value
    with client.write(shard_ref) as tensor:
        tensor.fill_(5.0)

    # Nested reads should work (shared locks allow multiple readers)
    with client.read(shard_ref) as tensor1:
        val1 = tensor1.sum().item()
        with client.read(shard_ref) as tensor2:
            val2 = tensor2.sum().item()
            assert val1 == val2 == 80.0, "Both reads should see same value"

    client.disconnect()


@pytest.mark.gpu
def test_multiple_tensors_same_client(infrastructure):
    """Test registering and using multiple tensors from the same client."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))

    # Register multiple tensors
    shard_refs = []
    for i in range(3):
        dims = [
            TensorDimSpec("dim_0", 4 * (i + 1), 0, 4 * (i + 1)),
            TensorDimSpec("dim_1", 8, 0, 8),
        ]
        shard_spec = TensorShardSpec(
            name=f"test_multi_tensor_{i}",
            dims=dims,
            dtype=torch.float32,
            device=device,
        )
        shard_ref = client.register_tensor_shard(shard_spec)
        assert shard_ref is not None
        shard_refs.append(shard_ref)

    # Write different values to each tensor
    for i, shard_ref in enumerate(shard_refs):
        with client.write(shard_ref) as tensor:
            tensor.fill_(float(i + 1))

    # Read and verify each tensor
    for i, shard_ref in enumerate(shard_refs):
        with client.read(shard_ref) as tensor:
            expected_shape = (4 * (i + 1), 8)
            assert tensor.shape == expected_shape, f"Tensor {i} shape mismatch"
            expected_val = float(i + 1)
            assert torch.allclose(
                tensor,
                torch.full(expected_shape, expected_val, device="cuda:0"),
            ), f"Tensor {i} value mismatch"

    client.disconnect()


@pytest.mark.gpu
def test_client_context_manager_pattern(infrastructure):
    """Test using client in a more manual pattern with explicit connect/disconnect."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    # Client connection
    client = Client(client_endpoint)
    assert client.is_connected, "Client should be connected"

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 4, 0, 4),
        TensorDimSpec("dim_1", 4, 0, 4),
    ]
    shard_spec = TensorShardSpec(
        name="test_ctx_pattern",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    with client.write(shard_ref) as tensor:
        tensor.fill_(99.0)

    with client.read(shard_ref) as tensor:
        assert tensor[0, 0].item() == 99.0

    # Explicit disconnect
    client.disconnect()
    assert not client.is_connected, "Client should be disconnected"


def _reader_process(
    endpoint: str,
    shard_ref,
    expected_value: float,
    result_queue,
    barrier,
):
    """Process that reads a tensor and verifies its value."""
    from setu.client import Client

    try:
        client = Client(endpoint)

        # Wait for all readers to be ready
        barrier.wait()

        # All readers attempt to read simultaneously
        with client.read(shard_ref) as tensor:
            # Hold the read lock for a bit to ensure overlap
            time.sleep(0.1)
            value = tensor.mean().item()
            matches = abs(value - expected_value) < 0.01

        client.disconnect()
        result_queue.put(("success", matches, value))
    except Exception as e:
        result_queue.put(("error", str(e), None))


@pytest.mark.gpu
def test_multiple_concurrent_readers(infrastructure):
    """Test that multiple readers can access the same tensor concurrently."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 8, 0, 8),
        TensorDimSpec("dim_1", 8, 0, 8),
    ]
    shard_spec = TensorShardSpec(
        name="test_concurrent_readers",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    # Write a known value
    expected_value = 7.5
    with client.write(shard_ref) as tensor:
        tensor.fill_(expected_value)

    client.disconnect()

    # Spawn multiple reader processes
    num_readers = 4
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    barrier = ctx.Barrier(num_readers)

    reader_procs = []
    for _ in range(num_readers):
        proc = ctx.Process(
            target=_reader_process,
            args=(
                client_endpoint,
                shard_ref,
                expected_value,
                result_queue,
                barrier,
            ),
        )
        proc.start()
        reader_procs.append(proc)

    # Wait for all readers to complete
    for proc in reader_procs:
        proc.join(timeout=10)

    # Check results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    assert (
        len(results) == num_readers
    ), f"Expected {num_readers} results, got {len(results)}"

    for status, matches, value in results:
        assert status == "success", f"Reader failed with error: {matches}"
        assert (
            matches
        ), f"Reader got unexpected value: {value}, expected {expected_value}"


def _writer_process(
    endpoint: str,
    shard_ref,
    write_value: float,
    result_queue,
    start_event,
):
    """Process that writes to a tensor."""
    from setu.client import Client

    try:
        client = Client(endpoint)

        # Wait for signal to start
        start_event.wait()

        start_time = time.time()
        with client.write(shard_ref) as tensor:
            # Hold the write lock for a bit
            time.sleep(0.2)
            tensor.fill_(write_value)
        elapsed = time.time() - start_time

        client.disconnect()
        result_queue.put(("success", elapsed, write_value))
    except Exception as e:
        result_queue.put(("error", str(e), None))


@pytest.mark.gpu
def test_writer_blocks_readers(infrastructure):
    """Test that a writer blocks readers (exclusive lock)."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 4, 0, 4),
        TensorDimSpec("dim_1", 4, 0, 4),
    ]
    shard_spec = TensorShardSpec(
        name="test_writer_blocks",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    # Write initial value
    with client.write(shard_ref) as tensor:
        tensor.fill_(1.0)

    client.disconnect()

    ctx = mp.get_context("spawn")
    writer_queue = ctx.Queue()
    reader_queue = ctx.Queue()
    start_event = ctx.Event()

    # Start writer first - it will wait for start_event
    writer_proc = ctx.Process(
        target=_writer_process,
        args=(
            client_endpoint,
            shard_ref,
            99.0,
            writer_queue,
            start_event,
        ),
    )
    writer_proc.start()

    # Give writer time to be ready
    time.sleep(0.1)

    # Start reader - uses a simple barrier that just waits for event
    reader_barrier = ctx.Barrier(1)  # Single reader, barrier is instant
    reader_proc = ctx.Process(
        target=_reader_process,
        args=(
            client_endpoint,
            shard_ref,
            99.0,  # Should see written value
            reader_queue,
            reader_barrier,
        ),
    )

    # Signal both to start
    start_event.set()
    time.sleep(0.05)  # Small delay to let writer acquire lock first
    reader_proc.start()

    writer_proc.join(timeout=10)
    reader_proc.join(timeout=10)

    # Writer should have completed
    writer_result = writer_queue.get()
    assert writer_result[0] == "success", f"Writer failed: {writer_result[1]}"

    # Reader should have seen the written value (waited for writer)
    reader_result = reader_queue.get()
    assert reader_result[0] == "success", f"Reader failed: {reader_result[1]}"
    assert reader_result[
        1
    ], f"Reader should see written value 99.0, got {reader_result[2]}"


@pytest.mark.gpu
def test_tensor_slice_operations(infrastructure):
    """Test performing slice operations on tensors within handles."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client_endpoint = infrastructure["client_endpoint"]

    client = Client(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("dim_0", 16, 0, 16),
        TensorDimSpec("dim_1", 32, 0, 32),
    ]
    shard_spec = TensorShardSpec(
        name="test_slice_ops",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None

    # Write using slices
    with client.write(shard_ref) as tensor:
        tensor.zero_()
        tensor[0:4, :] = 1.0
        tensor[4:8, :] = 2.0
        tensor[8:12, :] = 3.0
        tensor[12:16, :] = 4.0

    # Read and verify slices
    with client.read(shard_ref) as tensor:
        assert tensor[0:4, :].mean().item() == 1.0
        assert tensor[4:8, :].mean().item() == 2.0
        assert tensor[8:12, :].mean().item() == 3.0
        assert tensor[12:16, :].mean().item() == 4.0

    client.disconnect()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pytest.main([__file__, "-v", "--gpu"])
