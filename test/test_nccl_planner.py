"""
Tests for Planner compilation with NCCL backend.
"""

import uuid

import torch

from setu._commons.datatypes import (
    CopySpec,
    Device,
    TensorDim,
    TensorDimSpec,
    TensorSelection,
    TensorShardSpec,
)
from setu._coordinator import (
    Link,
    MetaStore,
    NCCLBackend,
    Participant,
    PassManager,
    Planner,
    RegisterSet,
    ShortestPathRouting,
    Topology,
)


def test_nccl_planner_no_optim_basic_copy():
    backend = NCCLBackend()
    pass_manager = PassManager()
    assert pass_manager.num_passes() == 0

    planner = Planner(backend, pass_manager)
    metastore = MetaStore()

    node_id = uuid.uuid4()
    device = Device(torch.device("cpu"))

    dim_spec = TensorDimSpec("dim0", 128, 0, 128)

    src_shard_spec = TensorShardSpec("tensor_a", [dim_spec], torch.float32, device)
    dst_shard_spec = TensorShardSpec("tensor_b", [dim_spec], torch.float32, device)

    metastore.register_tensor_shard(src_shard_spec, node_id)
    metastore.register_tensor_shard(dst_shard_spec, node_id)

    dim_map = {"dim0": TensorDim("dim0", 128)}
    src_selection = TensorSelection("tensor_a", dim_map)
    dst_selection = TensorSelection("tensor_b", dim_map)

    copy_spec = CopySpec("tensor_a", "tensor_b", src_selection, dst_selection)

    plan = planner.compile(copy_spec, metastore)

    assert plan is not None
    print(plan)


def test_nccl_planner_shortest_path_routing():
    n0 = uuid.UUID("01234567-89ab-cdef-0123-456789abcdef")
    n1 = uuid.UUID("00234567-89ab-cdef-0123-456789abcdef")

    dev0 = Device(torch.device("cuda:0"))
    dev1 = Device(torch.device("cuda:1"))

    # Build topology
    #   (n0,gpu0) <-> (n0,gpu1)  : latency=0,  bw=200
    #   (n0,gpu0) <-> (n1,gpu0)  : latency=10, bw=100
    #   (n0,gpu0) <-> (n1,gpu1)  : latency=10, bw=50
    #   (n0,gpu1) <-> (n1,gpu0)  : latency=10, bw=50
    #   (n0,gpu1) <-> (n1,gpu1)  : latency=10, bw=100
    topo = Topology()
    p_n0_g0 = Participant(n0, dev0)
    p_n0_g1 = Participant(n0, dev1)
    p_n1_g0 = Participant(n1, dev0)
    p_n1_g1 = Participant(n1, dev1)

    topo.add_bidirectional_link(p_n0_g0, p_n0_g1, Link(0, 200))
    topo.add_bidirectional_link(p_n0_g0, p_n1_g0, Link(10, 100))
    topo.add_bidirectional_link(p_n0_g0, p_n1_g1, Link(10, 50))
    topo.add_bidirectional_link(p_n0_g1, p_n1_g0, Link(10, 50))
    topo.add_bidirectional_link(p_n0_g1, p_n1_g1, Link(10, 100))

    pass_manager = PassManager()
    pass_manager.add_pass(ShortestPathRouting(topo))
    assert pass_manager.num_passes() == 1

    # 1MB single register per device
    reg = RegisterSet.uniform(4, 1024 * 1024)  # 4 registers of 1 MB each
    register_sets = {
        p_n0_g0: reg,
        p_n0_g1: reg,
        p_n1_g0: reg,
        p_n1_g1: reg,
    }
    backend = NCCLBackend(register_sets)
    planner = Planner(backend, pass_manager)

    metastore = MetaStore()

    dim_spec_0 = TensorDimSpec("dim0", 512, 0, 256)
    dim_spec_1 = TensorDimSpec("dim0", 512, 256, 512)

    src_shard_0 = TensorShardSpec("tensor_src", [dim_spec_0], torch.float16, dev0)
    src_shard_1 = TensorShardSpec("tensor_src", [dim_spec_1], torch.float16, dev0)
    dst_shard_0 = TensorShardSpec("tensor_dst", [dim_spec_0], torch.float16, dev0)
    dst_shard_1 = TensorShardSpec("tensor_dst", [dim_spec_1], torch.float16, dev1)

    metastore.register_tensor_shard(src_shard_0, n0)
    metastore.register_tensor_shard(src_shard_1, n0)
    metastore.register_tensor_shard(dst_shard_0, n1)
    metastore.register_tensor_shard(dst_shard_1, n1)

    dim_map = {"dim0": TensorDim("dim0", 512)}
    src_selection = TensorSelection("tensor_src", dim_map)
    dst_selection = TensorSelection("tensor_dst", dim_map)

    copy_spec = CopySpec("tensor_src", "tensor_dst", src_selection, dst_selection)

    plan = planner.compile(copy_spec, metastore)

    assert plan is not None
    print(plan)
