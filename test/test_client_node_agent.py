"""
Unit tests for Client-NodeAgent interaction.

Tests the communication flow between Client and NodeAgent components.
"""

import time
import uuid
from test.fixtures import ClusterSpec, DeviceSpec, SetuTestCluster

import pytest
import torch
import torch.multiprocessing as mp

from setu._commons.datatypes import Device


def _register_tensor(
    endpoint: str,
    tensor_name: str,
    dims_spec=None,
    node_id=None,
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

    tensor_ipc_spec, metadata, lock_base_dir = client.get_tensor_handle(shard_ref)
    client.disconnect()

    return tensor_ipc_spec


@pytest.fixture(scope="module")
def infrastructure():
    """
    Start Coordinator and NodeAgent once for all tests in this module.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    node_id = uuid.uuid4()
    spec = ClusterSpec(
        coordinator_port=29000,
        nodes={node_id: (29100, [DeviceSpec(Device(torch.device("cuda:0")))])},
    )

    with SetuTestCluster(spec) as cluster:
        yield {
            "client_endpoint": cluster.client_endpoint(node_id),
        }


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


@pytest.mark.gpu
def test_get_shards_tracking(infrastructure):
    """Test that client tracks registered shards via get_shards()."""
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    client_endpoint = infrastructure["client_endpoint"]

    client = Client()
    client.connect(client_endpoint)

    # Initially no shards
    assert len(client.get_shards()) == 0, "New client should have no shards"

    # Register first shard
    device = Device(torch_device=torch.device("cuda:0"))
    dims_1 = [
        TensorDimSpec("dim_0", 16, 0, 16),
        TensorDimSpec("dim_1", 32, 0, 32),
    ]
    shard_spec_1 = TensorShardSpec(
        name="tracked_tensor_1",
        dims=dims_1,
        dtype=torch.float32,
        device=device,
    )
    shard_ref_1 = client.register_tensor_shard(shard_spec_1)
    assert shard_ref_1 is not None

    # Should have 1 shard now
    shards = client.get_shards()
    assert len(shards) == 1, f"Expected 1 shard, got {len(shards)}"
    assert shards[0].name == "tracked_tensor_1"
    assert shards[0].shard_id == shard_ref_1.shard_id

    # Register second shard
    dims_2 = [
        TensorDimSpec("dim_0", 8, 0, 8),
        TensorDimSpec("dim_1", 16, 0, 16),
    ]
    shard_spec_2 = TensorShardSpec(
        name="tracked_tensor_2",
        dims=dims_2,
        dtype=torch.float32,
        device=device,
    )
    shard_ref_2 = client.register_tensor_shard(shard_spec_2)
    assert shard_ref_2 is not None

    # Should have 2 shards now
    shards = client.get_shards()
    assert len(shards) == 2, f"Expected 2 shards, got {len(shards)}"

    # Verify both shards are tracked
    shard_names = {s.name for s in shards}
    assert shard_names == {"tracked_tensor_1", "tracked_tensor_2"}

    client.disconnect()


@pytest.mark.gpu
def test_multiple_shards_same_client_get_handles(infrastructure):
    """Test registering multiple shards from same client and getting handles for all."""
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    client_endpoint = infrastructure["client_endpoint"]

    client = Client()
    client.connect(client_endpoint)

    device = Device(torch_device=torch.device("cuda:0"))

    # Register 3 different tensors
    tensor_configs = [
        ("multi_shard_a", [(4, 4), (8, 8)]),
        ("multi_shard_b", [(16, 16), (32, 32)]),
        ("multi_shard_c", [(2, 2), (4, 4), (8, 8)]),
    ]

    registered_refs = []
    for tensor_name, dim_sizes in tensor_configs:
        dims = [
            TensorDimSpec(f"dim_{i}", size, 0, size)
            for i, (_, size) in enumerate(dim_sizes)
        ]
        shard_spec = TensorShardSpec(
            name=tensor_name,
            dims=dims,
            dtype=torch.float32,
            device=device,
        )
        shard_ref = client.register_tensor_shard(shard_spec)
        assert shard_ref is not None, f"Failed to register {tensor_name}"
        registered_refs.append(shard_ref)

    # Verify all shards are tracked
    shards = client.get_shards()
    assert len(shards) == 3, f"Expected 3 shards, got {len(shards)}"

    # Get handles for all registered shards
    for shard_ref, (tensor_name, dim_sizes) in zip(registered_refs, tensor_configs):
        tensor_ipc_spec, metadata, lock_base_dir = client.get_tensor_handle(shard_ref)
        assert tensor_ipc_spec is not None, f"Failed to get handle for {tensor_name}"

        spec_dict = tensor_ipc_spec.to_dict()
        expected_sizes = [size for _, size in dim_sizes]
        assert spec_dict["tensor_size"] == expected_sizes, (
            f"Size mismatch for {tensor_name}: "
            f"expected {expected_sizes}, got {spec_dict['tensor_size']}"
        )

    client.disconnect()


@pytest.fixture(scope="module")
def multi_node_infrastructure():
    """Start Coordinator and two NodeAgents for distributed tensor tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    node_id_0 = uuid.uuid4()
    node_id_1 = uuid.uuid4()

    spec = ClusterSpec(
        coordinator_port=29200,
        nodes={
            node_id_0: (29201, [DeviceSpec(Device(torch.device("cuda:0")))]),
            node_id_1: (29202, [DeviceSpec(Device(torch.device("cuda:1")))]),
        },
    )

    with SetuTestCluster(spec) as cluster:
        yield {
            "client_endpoint_0": cluster.client_endpoint(node_id_0),
            "client_endpoint_1": cluster.client_endpoint(node_id_1),
            "node_id_0": node_id_0,
            "node_id_1": node_id_1,
        }


@pytest.mark.gpu
def test_register_overlapping_shard_returns_none(multi_node_infrastructure):
    """Test that registering overlapping shards returns None."""
    from setu._commons.datatypes import TensorDimSpec

    infra = multi_node_infrastructure
    tensor_name = "overlapping_tensor"

    # Register first shard covering rows [0, 600)
    dims_0 = [
        TensorDimSpec("rows", 1024, 0, 600),
        TensorDimSpec("cols", 768, 0, 768),
    ]
    shard_ref_0 = _register_tensor(
        infra["client_endpoint_0"],
        tensor_name,
        dims_0,
        infra["node_id_0"],
        0,
    )
    assert shard_ref_0 is not None, "First shard should register successfully"

    # Attempt to register overlapping shard covering rows [400, 1024)
    # This overlaps with [400, 600)
    dims_1 = [
        TensorDimSpec("rows", 1024, 400, 1024),
        TensorDimSpec("cols", 768, 0, 768),
    ]
    shard_ref_1 = _register_tensor(
        infra["client_endpoint_1"],
        tensor_name,
        dims_1,
        infra["node_id_1"],
        1,
    )
    assert shard_ref_1 is None, "Overlapping shard should return None"


@pytest.mark.gpu
def test_register_dtype_mismatch_returns_none(multi_node_infrastructure):
    """Test that registering shards with mismatched dtypes returns None."""
    from setu._client import Client
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec

    infra = multi_node_infrastructure
    tensor_name = "dtype_mismatch_tensor"

    # Register first shard with float32
    dims = [
        TensorDimSpec("rows", 1024, 0, 512),
        TensorDimSpec("cols", 768, 0, 768),
    ]

    client_0 = Client()
    client_0.connect(infra["client_endpoint_0"])
    device_0 = Device(torch_device=torch.device("cuda:0"))
    shard_spec_0 = TensorShardSpec(
        name=tensor_name,
        dims=dims,
        dtype=torch.float32,
        device=device_0,
    )
    shard_ref_0 = client_0.register_tensor_shard(shard_spec_0)
    client_0.disconnect()
    assert shard_ref_0 is not None, "First shard should register successfully"

    # Attempt to register second shard with float16 (dtype mismatch)
    dims_1 = [
        TensorDimSpec("rows", 1024, 512, 1024),
        TensorDimSpec("cols", 768, 0, 768),
    ]

    client_1 = Client()
    client_1.connect(infra["client_endpoint_1"])
    device_1 = Device(torch_device=torch.device("cuda:1"))
    shard_spec_1 = TensorShardSpec(
        name=tensor_name,
        dims=dims_1,
        dtype=torch.float16,  # Different dtype
        device=device_1,
    )
    shard_ref_1 = client_1.register_tensor_shard(shard_spec_1)
    client_1.disconnect()
    assert shard_ref_1 is None, "Dtype mismatch shard should return None"


@pytest.mark.gpu
def test_register_dimension_size_mismatch_returns_none(multi_node_infrastructure):
    """Test that registering shards with mismatched dimension sizes returns None."""
    from setu._commons.datatypes import TensorDimSpec

    infra = multi_node_infrastructure
    tensor_name = "dim_size_mismatch_tensor"

    # Register first shard with rows total size 1024
    dims_0 = [
        TensorDimSpec("rows", 1024, 0, 512),
        TensorDimSpec("cols", 768, 0, 768),
    ]
    shard_ref_0 = _register_tensor(
        infra["client_endpoint_0"],
        tensor_name,
        dims_0,
        infra["node_id_0"],
        0,
    )
    assert shard_ref_0 is not None, "First shard should register successfully"

    # Attempt to register shard with rows total size 2048 (size mismatch)
    dims_1 = [
        TensorDimSpec("rows", 2048, 512, 1024),  # Different total size
        TensorDimSpec("cols", 768, 0, 768),
    ]
    shard_ref_1 = _register_tensor(
        infra["client_endpoint_1"],
        tensor_name,
        dims_1,
        infra["node_id_1"],
        1,
    )
    assert shard_ref_1 is None, "Dimension size mismatch shard should return None"


@pytest.mark.gpu
def test_register_dimension_name_mismatch_returns_none(multi_node_infrastructure):
    """Test that registering shards with mismatched dimension names returns None."""
    from setu._commons.datatypes import TensorDimSpec

    infra = multi_node_infrastructure
    tensor_name = "dim_name_mismatch_tensor"

    # Register first shard with dimension names "rows" and "cols"
    dims_0 = [
        TensorDimSpec("rows", 1024, 0, 512),
        TensorDimSpec("cols", 768, 0, 768),
    ]
    shard_ref_0 = _register_tensor(
        infra["client_endpoint_0"],
        tensor_name,
        dims_0,
        infra["node_id_0"],
        0,
    )
    assert shard_ref_0 is not None, "First shard should register successfully"

    # Attempt to register shard with different dimension names
    dims_1 = [
        TensorDimSpec("height", 1024, 512, 1024),  # Different name
        TensorDimSpec("width", 768, 0, 768),  # Different name
    ]
    shard_ref_1 = _register_tensor(
        infra["client_endpoint_1"],
        tensor_name,
        dims_1,
        infra["node_id_1"],
        1,
    )
    assert shard_ref_1 is None, "Dimension name mismatch shard should return None"


@pytest.mark.gpu
def test_register_identical_shard_returns_none(multi_node_infrastructure):
    """Test that registering an identical (fully overlapping) shard returns None."""
    from setu._commons.datatypes import TensorDimSpec

    infra = multi_node_infrastructure
    tensor_name = "identical_shard_tensor"

    # Register first shard
    dims = [
        TensorDimSpec("rows", 1024, 0, 512),
        TensorDimSpec("cols", 768, 0, 768),
    ]
    shard_ref_0 = _register_tensor(
        infra["client_endpoint_0"],
        tensor_name,
        dims,
        infra["node_id_0"],
        0,
    )
    assert shard_ref_0 is not None, "First shard should register successfully"

    # Attempt to register identical shard (same ranges)
    shard_ref_1 = _register_tensor(
        infra["client_endpoint_1"],
        tensor_name,
        dims,  # Same dimensions and ranges
        infra["node_id_1"],
        1,
    )
    assert shard_ref_1 is None, "Identical shard should return None (fully overlapping)"


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
    )
    assert shard_ref_1 is not None

    # Wait for AllocateTensorRequest to be processed
    time.sleep(0.5)

    # Verify both NodeAgents allocated the tensor by getting handles
    client_0 = Client()
    client_0.connect(infra["client_endpoint_0"])
    ipc_spec_0, metadata_0, lock_dir_0 = client_0.get_tensor_handle(shard_ref_0)
    client_0.disconnect()

    client_1 = Client()
    client_1.connect(infra["client_endpoint_1"])
    ipc_spec_1, metadata_1, lock_dir_1 = client_1.get_tensor_handle(shard_ref_1)
    client_1.disconnect()

    assert ipc_spec_0 is not None, "NodeAgent 0 should have allocated tensor"
    assert ipc_spec_1 is not None, "NodeAgent 1 should have allocated tensor"

    assert ipc_spec_0.to_dict()["tensor_size"] == [512, 768]
    assert ipc_spec_1.to_dict()["tensor_size"] == [512, 768]


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pytest.main([__file__, "-v", "--gpu"])
