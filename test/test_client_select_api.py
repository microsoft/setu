"""
Integration tests for the Client.select() API and TensorSelection operations.

The select() API returns a spanning TensorSelection over the full tensor
dimensions (all indices selected). The selection can then be narrowed using
where() and combined using get_intersection().

Tests:
- select() returns spanning selection for the full tensor
- select() returns the same spanning selection regardless of shard ownership
- where() narrows selections correctly
- get_intersection() computes correct intersections
"""

import time
import uuid

import pytest
import torch
import torch.multiprocessing as mp


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


@pytest.fixture(scope="module")
def infrastructure():
    """Start Coordinator and NodeAgent for all tests in this module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    coordinator_port = 29600
    node_agent_port = 29601
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

    time.sleep(0.2)

    yield {"client_endpoint": client_endpoint}

    stop_event.set()
    time.sleep(0.3)
    for proc in [node_agent_proc, coordinator_proc]:
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)


def make_spanning_selection(name, dims):
    """
    Create a spanning TensorSelection with all indices selected.

    Args:
        name: Tensor name
        dims: Dict of {dim_name: total_size}

    Returns:
        Native TensorSelection with all indices selected (spanning)
    """
    from setu._commons.datatypes import TensorDim, TensorSelection

    dim_map = {dim_name: TensorDim(dim_name, size) for dim_name, size in dims.items()}
    return TensorSelection(name, dim_map)


# ==============================================================================
# Select Tests - Spanning Selection
# ==============================================================================


@pytest.mark.gpu
def test_select_full_shard_returns_spanning(infrastructure):
    """Test select() returns spanning selection for a fully-owned tensor."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 32, 0, 32),
        TensorDimSpec("cols", 64, 0, 64),
    ]
    spec = TensorShardSpec(
        name="select_full_spanning",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )

    client.register_tensor_shard(spec)

    actual = client.select("select_full_spanning")
    expected = make_spanning_selection(
        "select_full_spanning",
        {"rows": 32, "cols": 64},
    )

    assert actual.native == expected
    assert actual.is_spanning()

    client.disconnect()


@pytest.mark.gpu
def test_select_partial_shard_returns_spanning(infrastructure):
    """Test select() on a partial shard still returns spanning selection."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 100, 20, 50),  # Only rows 20-50 of 100
        TensorDimSpec("cols", 64, 0, 64),
    ]
    spec = TensorShardSpec(
        name="select_partial_spanning",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )

    client.register_tensor_shard(spec)

    actual = client.select("select_partial_spanning")
    expected = make_spanning_selection(
        "select_partial_spanning",
        {"rows": 100, "cols": 64},
    )

    # select() always returns spanning — not limited to owned range
    assert actual.native == expected
    assert actual.is_spanning()

    client.disconnect()


@pytest.mark.gpu
def test_select_3d_tensor_returns_spanning(infrastructure):
    """Test select() on 3D tensor returns spanning selection."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("batch", 16, 4, 12),
        TensorDimSpec("rows", 64, 0, 32),
        TensorDimSpec("cols", 128, 64, 128),
    ]
    spec = TensorShardSpec(
        name="select_3d_spanning",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )

    client.register_tensor_shard(spec)

    actual = client.select("select_3d_spanning")
    expected = make_spanning_selection(
        "select_3d_spanning",
        {"batch": 16, "rows": 64, "cols": 128},
    )

    assert actual.native == expected
    assert actual.is_spanning()

    client.disconnect()


# ==============================================================================
# Select Tests - Multiple Shards
# ==============================================================================


@pytest.mark.gpu
def test_select_multiple_shards_returns_spanning(infrastructure):
    """Test select() with multiple shards returns spanning selection."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))

    # First shard: rows 0-32
    dims1 = [
        TensorDimSpec("rows", 64, 0, 32),
        TensorDimSpec("cols", 32, 0, 32),
    ]
    spec1 = TensorShardSpec(
        name="select_multi_spanning",
        dims=dims1,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec1)

    # Second shard: rows 32-64
    dims2 = [
        TensorDimSpec("rows", 64, 32, 64),
        TensorDimSpec("cols", 32, 0, 32),
    ]
    spec2 = TensorShardSpec(
        name="select_multi_spanning",
        dims=dims2,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec2)

    actual = client.select("select_multi_spanning")
    expected = make_spanning_selection(
        "select_multi_spanning",
        {"rows": 64, "cols": 32},
    )

    assert actual.native == expected
    assert actual.is_spanning()

    client.disconnect()


@pytest.mark.gpu
def test_select_after_single_shard_same_as_after_all(infrastructure):
    """Test select() returns the same spanning selection whether called
    after one shard or after all shards are registered."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))

    # Register first shard
    dims1 = [
        TensorDimSpec("rows", 80, 0, 40),
        TensorDimSpec("cols", 50, 0, 50),
    ]
    spec1 = TensorShardSpec(
        name="select_same_spanning",
        dims=dims1,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec1)

    sel_after_first = client.select("select_same_spanning")

    # Register second shard
    dims2 = [
        TensorDimSpec("rows", 80, 40, 80),
        TensorDimSpec("cols", 50, 0, 50),
    ]
    spec2 = TensorShardSpec(
        name="select_same_spanning",
        dims=dims2,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec2)

    sel_after_second = client.select("select_same_spanning")

    # Both should be identical spanning selections
    assert sel_after_first.native == sel_after_second.native
    assert sel_after_first.is_spanning()
    assert sel_after_second.is_spanning()

    client.disconnect()


# ==============================================================================
# TensorSelection where() Tests
# ==============================================================================


@pytest.mark.gpu
def test_select_where_narrows_selection(infrastructure):
    """Test where() narrows a spanning selection correctly."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 32, 0, 32),
        TensorDimSpec("cols", 64, 0, 64),
    ]
    spec = TensorShardSpec(
        name="where_narrow_spanning",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec)

    base = client.select("where_narrow_spanning")
    assert base.is_spanning()

    narrowed = base.where("rows", [0, 5, 10, 15])

    expected = make_spanning_selection(
        "where_narrow_spanning",
        {"rows": 32, "cols": 64},
    ).where("rows", {0, 5, 10, 15})

    assert narrowed.native == expected
    assert not narrowed.is_spanning()

    client.disconnect()


@pytest.mark.gpu
def test_select_where_multiple_dims(infrastructure):
    """Test where() narrowing on multiple dimensions."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 32, 0, 32),
        TensorDimSpec("cols", 64, 0, 64),
    ]
    spec = TensorShardSpec(
        name="where_multi_dim",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec)

    narrowed = client.select("where_multi_dim").where("rows", [0, 1, 2]).where(
        "cols", [10, 20, 30]
    )

    expected = (
        make_spanning_selection(
            "where_multi_dim",
            {"rows": 32, "cols": 64},
        )
        .where("rows", {0, 1, 2})
        .where("cols", {10, 20, 30})
    )

    assert narrowed.native == expected
    assert not narrowed.is_spanning()

    client.disconnect()


@pytest.mark.gpu
def test_select_where_disjoint_intersection_is_empty(infrastructure):
    """Test where() intersections with disjoint sets produce empty selection."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 32, 0, 32),
        TensorDimSpec("cols", 50, 0, 50),
    ]
    spec = TensorShardSpec(
        name="where_empty",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec)

    # Narrow to rows {0,1,2}, then intersect with rows {10,11,12} — disjoint
    sel1 = client.select("where_empty").where("rows", [0, 1, 2])
    sel2 = client.select("where_empty").where("rows", [10, 11, 12])
    intersection = sel1.get_intersection(sel2)

    assert intersection.is_empty()

    client.disconnect()


# ==============================================================================
# TensorSelection intersection() Tests
# ==============================================================================


@pytest.mark.gpu
def test_select_intersection_matches_expected(infrastructure):
    """Test get_intersection() returns correct intersection."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 32, 0, 32),
        TensorDimSpec("cols", 32, 0, 32),
    ]
    spec = TensorShardSpec(
        name="intersection_spanning",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec)

    sel1 = client.select("intersection_spanning").where("rows", [0, 1, 2, 3, 4, 5])
    sel2 = client.select("intersection_spanning").where("rows", [3, 4, 5, 6, 7, 8])

    intersection = sel1.get_intersection(sel2)

    # Expected: rows {3, 4, 5}, cols all
    expected = make_spanning_selection(
        "intersection_spanning",
        {"rows": 32, "cols": 32},
    ).where("rows", {3, 4, 5})

    assert intersection.native == expected

    client.disconnect()


@pytest.mark.gpu
def test_select_intersection_disjoint_is_empty(infrastructure):
    """Test get_intersection() with disjoint selections is empty."""
    from setu._commons.datatypes import Device, TensorDimSpec, TensorShardSpec
    from setu.client import Client

    client = Client(infrastructure["client_endpoint"])

    device = Device(torch_device=torch.device("cuda:0"))
    dims = [
        TensorDimSpec("rows", 32, 0, 32),
        TensorDimSpec("cols", 32, 0, 32),
    ]
    spec = TensorShardSpec(
        name="intersection_disjoint",
        dims=dims,
        dtype=torch.float32,
        device=device,
    )
    client.register_tensor_shard(spec)

    sel1 = client.select("intersection_disjoint").where("rows", [0, 1, 2])
    sel2 = client.select("intersection_disjoint").where("rows", [10, 11, 12])

    intersection = sel1.get_intersection(sel2)
    assert intersection.is_empty()

    client.disconnect()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    pytest.main([__file__, "-v", "--gpu"])
