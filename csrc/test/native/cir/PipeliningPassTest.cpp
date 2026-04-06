//==============================================================================
// Copyright (c) 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================
#include <gtest/gtest.h>
//==============================================================================
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "commons/Types.h"
#include "planner/RegisterSet.h"
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
#include "planner/passes/PassContext.h"
#include "planner/passes/Pipelining.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::commons::NodeId;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::Value;
using setu::planner::passes::PassContext;
using setu::planner::passes::Pipelining;

Device MakePipeDevice(NodeId node_id, std::int16_t gpu_index = 0) {
  return Device(node_id, setu::commons::datatypes::Device(torch::Device(
                             torch::kCUDA, static_cast<int8_t>(gpu_index))));
}

NodeId MakePipeNodeId(std::string uuid_str) {
  boost::uuids::string_generator gen;
  return gen(uuid_str);
}

setu::planner::ir::ref::ShardRef MakePipeEmptyShardRef() {
  return setu::planner::ir::ref::ShardRef(boost::uuids::nil_uuid());
}

//==============================================================================
namespace {

class CIRPipeliningTest : public testing::Test {
 protected:
  CIRPipeliningTest() {}

  NodeId n0 = MakePipeNodeId("01234567-89ab-cdef-0123-456789abcdef");
  NodeId n1 = MakePipeNodeId("00234567-89ab-cdef-0123-456789abcdef");
  NodeId n2 = MakePipeNodeId("00034567-89ab-cdef-0123-456789abcdef");
  torch::Dtype dt = torch::kFloat16;
  std::unordered_map<Device, setu::planner::RegisterSet> empty_register_sets;
  setu::planner::hints::HintStore empty_hints;

  PassContext DefaultCtx() {
    return PassContext{.hints = empty_hints,
                       .register_sets = empty_register_sets};
  }

  /// Build a 2-hop relay program: src(devA) → tmp(devB) → dst(devC)
  /// with the given payload size.
  Program MakeTwoHopProgram(Device dev_a, Device dev_b, Device dev_c,
                            std::size_t payload_elements) {
    Program p;
    auto v_src = p.EmitView(dev_a, MakePipeEmptyShardRef(),
                            Slice{0, payload_elements}, dt);
    auto v_dst = p.EmitView(dev_c, MakePipeEmptyShardRef(),
                            Slice{0, payload_elements}, dt);
    auto tmp =
        p.EmitAllocTmp(dev_b, payload_elements * torch::elementSize(dt), dt);

    auto s_src = p.EmitSlice(v_src, Slice{0, payload_elements});
    auto s_dst = p.EmitSlice(v_dst, Slice{0, payload_elements});
    auto s_tmp = p.EmitSlice(tmp, Slice{0, payload_elements});

    auto c0 = p.EmitCopy(s_src, s_tmp);  // hop 1: A→B
    (void)p.EmitCopy(c0, s_dst);         // hop 2: B→C
    (void)p.EmitConsume(v_dst);
    return p;
  }
};

TEST_F(CIRPipeliningTest, SingleHop_Unchanged) {
  Program program;
  auto dev0 = MakePipeDevice(n0, 0);
  auto dev1 = MakePipeDevice(n1, 0);

  auto v0 = program.EmitView(dev0, MakePipeEmptyShardRef(), Slice{0, 256}, dt);
  auto v1 = program.EmitView(dev1, MakePipeEmptyShardRef(), Slice{0, 256}, dt);
  (void)program.EmitCopy(v0, v1);

  auto before = program.Dump();

  Pipelining pass(128);
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), before);
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(CIRPipeliningTest, TwoHop_PayloadFitsInOneChunk_Unchanged) {
  auto dev_a = MakePipeDevice(n0, 0);
  auto dev_b = MakePipeDevice(n0, 1);
  auto dev_c = MakePipeDevice(n1, 0);

  auto program = MakeTwoHopProgram(dev_a, dev_b, dev_c, 256);
  auto before = program.Dump();

  Pipelining pass(256);  // chunk_size == payload → no splitting
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), before);
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(CIRPipeliningTest, TwoHop_ChunkedWavefront) {
  auto dev_a = MakePipeDevice(n0, 0);
  auto dev_b = MakePipeDevice(n0, 1);
  auto dev_c = MakePipeDevice(n1, 0);

  auto program = MakeTwoHopProgram(dev_a, dev_b, dev_c, 512);

  Pipelining pass(256);  // 512 / 256 = 2 chunks
  auto result = pass.Run(std::move(program), DefaultCtx());

  // Wavefront order for 2 chunks, 2 hops (descending hop within diagonal):
  // micro_stage 0: (chunk0, hop0)
  // micro_stage 1: (chunk0, hop1), (chunk1, hop0)  ← send before receive
  // micro_stage 2: (chunk1, hop1)
  // Each hop emits: slice(src), slice(dst), copy
  // After all: consume
  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=01234567-89ab-cdef-0123-456789abcdef, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 512], Half)
  [1] %1 = view(Participant(node_id=00234567-89ab-cdef-0123-456789abcdef, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 512], Half)
  [2] %2 = alloc_tmp(Participant(node_id=01234567-89ab-cdef-0123-456789abcdef, device=Device(torch_device=cuda:1)), 1024, Half)
  [3] %3 = slice(%0, [0, 512])
  [4] %4 = slice(%1, [0, 512])
  [5] %5 = slice(%2, [0, 512])
  [6] %6 = slice(%3, [0, 256])
  [7] %7 = slice(%5, [0, 256])
  [8] %8 = copy(%6, %7)
  [9] %9 = slice(%4, [0, 256])
  [10] %10 = copy(%8, %9)
  [11] %11 = slice(%3, [256, 256])
  [12] %12 = slice(%5, [256, 256])
  [13] %13 = copy(%11, %12)
  [14] %14 = slice(%4, [256, 256])
  [15] %15 = copy(%13, %14)
  [16] %16 = consume(%4)
  [17] %17 = consume(%1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(CIRPipeliningTest, ThreeHop_WavefrontOrder) {
  auto dev_a = MakePipeDevice(n0, 0);
  auto dev_b = MakePipeDevice(n0, 1);
  auto dev_c = MakePipeDevice(n1, 0);
  auto dev_d = MakePipeDevice(n2, 0);

  // Build 3-hop relay: A → B → C → D
  Program program;
  std::size_t payload = 512;
  auto v_src =
      program.EmitView(dev_a, MakePipeEmptyShardRef(), Slice{0, payload}, dt);
  auto v_dst =
      program.EmitView(dev_d, MakePipeEmptyShardRef(), Slice{0, payload}, dt);
  auto tmp_b =
      program.EmitAllocTmp(dev_b, payload * torch::elementSize(dt), dt);
  auto tmp_c =
      program.EmitAllocTmp(dev_c, payload * torch::elementSize(dt), dt);

  auto s_src = program.EmitSlice(v_src, Slice{0, payload});
  auto s_dst = program.EmitSlice(v_dst, Slice{0, payload});
  auto s_tmp_b = program.EmitSlice(tmp_b, Slice{0, payload});
  auto s_tmp_c = program.EmitSlice(tmp_c, Slice{0, payload});

  auto c0 = program.EmitCopy(s_src, s_tmp_b);  // hop 1: A→B
  auto c1 = program.EmitCopy(c0, s_tmp_c);     // hop 2: B→C
  (void)program.EmitCopy(c1, s_dst);           // hop 3: C→D
  (void)program.EmitConsume(v_dst);

  Pipelining pass(256);  // 2 chunks, 3 hops → 4 micro-stages
  auto result = pass.Run(std::move(program), DefaultCtx());

  // Verify linearity and basic structure
  EXPECT_NO_THROW(Linearity::Check(result));

  // Verify wavefront order by checking copy depth analysis
  auto copy_depth = setu::planner::ir::cir::CopyDepthAnalysis::Build(result);

  // Collect copies in program order with their depths
  std::vector<std::uint32_t> copy_depths;
  for (std::uint32_t op_idx = 0; op_idx < result.NumOperations(); ++op_idx) {
    if (result.Operations()[op_idx].Type() ==
        setu::planner::ir::cir::OpType::kCopy) {
      ASSERT_TRUE(copy_depth.depth[op_idx].has_value());
      copy_depths.push_back(copy_depth.depth[op_idx].value());
    }
  }

  // 2 chunks × 3 hops = 6 copies
  // Wavefront order (descending hop within diagonal):
  //   s=0: (hop0,c0)
  //   s=1: (hop1,c0), (hop0,c1)   ← send before receive
  //   s=2: (hop2,c0), (hop1,c1)   ← send before receive
  //   s=3: (hop2,c1)
  // Depths:  0, 1, 0, 2, 1, 2
  ASSERT_EQ(copy_depths.size(), 6);
  EXPECT_EQ(copy_depths[0], 0);  // hop0 chunk0
  EXPECT_EQ(copy_depths[1], 1);  // hop1 chunk0
  EXPECT_EQ(copy_depths[2], 0);  // hop0 chunk1
  EXPECT_EQ(copy_depths[3], 2);  // hop2 chunk0
  EXPECT_EQ(copy_depths[4], 1);  // hop1 chunk1
  EXPECT_EQ(copy_depths[5], 2);  // hop2 chunk1
}

TEST_F(CIRPipeliningTest, MultipleIndependentChains) {
  auto dev_a = MakePipeDevice(n0, 0);
  auto dev_b = MakePipeDevice(n0, 1);
  auto dev_c = MakePipeDevice(n1, 0);

  Program program;
  std::size_t payload = 512;

  // Chain 1: src1(A) → tmp1(B) → dst1(C)
  auto v_src1 =
      program.EmitView(dev_a, MakePipeEmptyShardRef(), Slice{0, payload}, dt);
  auto v_dst1 =
      program.EmitView(dev_c, MakePipeEmptyShardRef(), Slice{0, payload}, dt);
  auto tmp1 = program.EmitAllocTmp(dev_b, payload * torch::elementSize(dt), dt);
  auto s_src1 = program.EmitSlice(v_src1, Slice{0, payload});
  auto s_dst1 = program.EmitSlice(v_dst1, Slice{0, payload});
  auto s_tmp1 = program.EmitSlice(tmp1, Slice{0, payload});
  auto c0_1 = program.EmitCopy(s_src1, s_tmp1);
  (void)program.EmitCopy(c0_1, s_dst1);
  (void)program.EmitConsume(v_dst1);

  // Chain 2: src2(A) → tmp2(B) → dst2(C)
  auto v_src2 = program.EmitView(dev_a, MakePipeEmptyShardRef(),
                                 Slice{payload, payload}, dt);
  auto v_dst2 = program.EmitView(dev_c, MakePipeEmptyShardRef(),
                                 Slice{payload, payload}, dt);
  auto tmp2 = program.EmitAllocTmp(dev_b, payload * torch::elementSize(dt), dt);
  auto s_src2 = program.EmitSlice(v_src2, Slice{0, payload});
  auto s_dst2 = program.EmitSlice(v_dst2, Slice{0, payload});
  auto s_tmp2 = program.EmitSlice(tmp2, Slice{0, payload});
  auto c0_2 = program.EmitCopy(s_src2, s_tmp2);
  (void)program.EmitCopy(c0_2, s_dst2);
  (void)program.EmitConsume(v_dst2);

  Pipelining pass(256);
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_NO_THROW(Linearity::Check(result));

  // Each chain produces 4 copies (2 chunks × 2 hops) = 8 total
  std::uint32_t copy_count = 0;
  for (std::uint32_t i = 0; i < result.NumOperations(); ++i) {
    if (result.Operations()[i].Type() ==
        setu::planner::ir::cir::OpType::kCopy) {
      copy_count++;
    }
  }
  EXPECT_EQ(copy_count, 8);
}

TEST_F(CIRPipeliningTest, TwoHop_RelayDeviceSendFollowsReceivePerChunk) {
  // Efficiency test: on the relay device, each chunk's send (hop1) must appear
  // immediately after its receive (hop0) — no batching of receives.
  auto dev_a = MakePipeDevice(n0, 0);
  auto dev_b = MakePipeDevice(n0, 1);  // relay
  auto dev_c = MakePipeDevice(n1, 0);

  auto program = MakeTwoHopProgram(dev_a, dev_b, dev_c, 1024);

  Pipelining pass(256);  // 4 chunks, 2 hops
  auto result = pass.Run(std::move(program), DefaultCtx());
  EXPECT_NO_THROW(Linearity::Check(result));

  // Collect copy depths in program order.
  auto copy_depth = setu::planner::ir::cir::CopyDepthAnalysis::Build(result);

  std::vector<std::uint32_t> depths;
  for (std::uint32_t i = 0; i < result.NumOperations(); ++i) {
    if (result.Operations()[i].Type() ==
        setu::planner::ir::cir::OpType::kCopy) {
      ASSERT_TRUE(copy_depth.depth[i].has_value());
      depths.push_back(copy_depth.depth[i].value());
    }
  }

  // 4 chunks × 2 hops = 8 copies.
  // Descending hop order: after the first receive (depth 0), every subsequent
  // pair should be (depth 1, depth 0) — send then receive — except the last
  // which is just depth 1.
  // Expected: 0, 1, 0, 1, 0, 1, 0, 1
  ASSERT_EQ(depths.size(), 8);
  for (std::size_t i = 0; i < depths.size(); ++i) {
    EXPECT_EQ(depths[i], i % 2) << "Copy " << i << " has wrong depth";
  }
}

TEST_F(CIRPipeliningTest, ThreeHop_RelayDeviceSendFollowsReceivePerChunk) {
  // Efficiency test for 3-hop relay with enough chunks to see steady state.
  auto dev_a = MakePipeDevice(n0, 0);
  auto dev_b = MakePipeDevice(n0, 1);
  auto dev_c = MakePipeDevice(n1, 0);
  auto dev_d = MakePipeDevice(n2, 0);

  Program program;
  std::size_t payload = 1024;
  auto v_src =
      program.EmitView(dev_a, MakePipeEmptyShardRef(), Slice{0, payload}, dt);
  auto v_dst =
      program.EmitView(dev_d, MakePipeEmptyShardRef(), Slice{0, payload}, dt);
  auto tmp_b =
      program.EmitAllocTmp(dev_b, payload * torch::elementSize(dt), dt);
  auto tmp_c =
      program.EmitAllocTmp(dev_c, payload * torch::elementSize(dt), dt);

  auto s_src = program.EmitSlice(v_src, Slice{0, payload});
  auto s_dst = program.EmitSlice(v_dst, Slice{0, payload});
  auto s_tmp_b = program.EmitSlice(tmp_b, Slice{0, payload});
  auto s_tmp_c = program.EmitSlice(tmp_c, Slice{0, payload});

  auto c0 = program.EmitCopy(s_src, s_tmp_b);
  auto c1 = program.EmitCopy(c0, s_tmp_c);
  (void)program.EmitCopy(c1, s_dst);
  (void)program.EmitConsume(v_dst);

  Pipelining pass(256);  // 4 chunks, 3 hops
  auto result = pass.Run(std::move(program), DefaultCtx());
  EXPECT_NO_THROW(Linearity::Check(result));

  auto copy_depth = setu::planner::ir::cir::CopyDepthAnalysis::Build(result);

  std::vector<std::uint32_t> depths;
  for (std::uint32_t i = 0; i < result.NumOperations(); ++i) {
    if (result.Operations()[i].Type() ==
        setu::planner::ir::cir::OpType::kCopy) {
      ASSERT_TRUE(copy_depth.depth[i].has_value());
      depths.push_back(copy_depth.depth[i].value());
    }
  }

  // 4 chunks × 3 hops = 12 copies.  Descending hop within diagonal:
  //   s=0: (hop0,c0)                       depths: 0
  //   s=1: (hop1,c0), (hop0,c1)                    1, 0
  //   s=2: (hop2,c0), (hop1,c1), (hop0,c2)         2, 1, 0
  //   s=3: (hop2,c1), (hop1,c2), (hop0,c3)         2, 1, 0
  //   s=4: (hop2,c2), (hop1,c3)                     2, 1
  //   s=5: (hop2,c3)                                2
  ASSERT_EQ(depths.size(), 12);
  std::vector<std::uint32_t> expected = {0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2};
  for (std::size_t i = 0; i < depths.size(); ++i) {
    EXPECT_EQ(depths[i], expected[i]) << "Copy " << i << " has wrong depth";
  }
}

TEST_F(CIRPipeliningTest, Idempotent) {
  auto dev_a = MakePipeDevice(n0, 0);
  auto dev_b = MakePipeDevice(n0, 1);
  auto dev_c = MakePipeDevice(n1, 0);

  auto program = MakeTwoHopProgram(dev_a, dev_b, dev_c, 512);

  Pipelining pass(256);
  auto result1 = pass.Run(std::move(program), DefaultCtx());
  auto dump1 = result1.Dump();

  auto result2 = pass.Run(std::move(result1), DefaultCtx());
  auto dump2 = result2.Dump();

  EXPECT_EQ(dump1, dump2);
  EXPECT_NO_THROW(Linearity::Check(result2));
}

//==============================================================================
// Pack → Copy → Unpack pipelining tests
//==============================================================================

/// Build a Pack → Copy → Unpack program (mimics PackUnpackCopies output).
///
/// Creates N source views on src_dev, N destination views on dst_dev,
/// then: alloc_tmp(src) → pack(srcs, tmp) → alloc_tmp(dst) →
///       copy(packed, tmp) → unpack(copied, dsts).
Program MakePackCopyUnpackProgram(
    Device src_dev, Device dst_dev, torch::Dtype dtype,
    const std::vector<std::size_t>& piece_sizes) {
  Program p;
  auto shard = MakePipeEmptyShardRef();

  std::vector<Value> src_views;
  std::vector<Value> dst_views;
  std::size_t total_elements = 0;

  for (std::size_t piece_size : piece_sizes) {
    src_views.push_back(p.EmitView(src_dev, shard, Slice{0, piece_size}, dtype));
    dst_views.push_back(p.EmitView(dst_dev, shard, Slice{0, piece_size}, dtype));
    total_elements += piece_size;
  }

  auto src_tmp = p.EmitAllocTmp(src_dev, total_elements, dtype);
  auto packed = p.EmitPack(src_views, src_tmp);

  auto dst_tmp = p.EmitAllocTmp(dst_dev, total_elements, dtype);
  auto copied = p.EmitCopy(packed, dst_tmp);

  (void)p.EmitUnpack(copied, dst_views);
  return p;
}

TEST_F(CIRPipeliningTest, PackCopyUnpack_SmallPayload_Unchanged) {
  auto dev0 = MakePipeDevice(n0, 0);
  auto dev1 = MakePipeDevice(n1, 0);

  // 4 pieces × 64 elements = 256 total.  Chunk size = 512 elements.
  // Everything fits in one chunk → no pipelining.
  auto program = MakePackCopyUnpackProgram(dev0, dev1, dt, {64, 64, 64, 64});
  auto before = program.Dump();

  Pipelining pass(512 * torch::elementSize(dt));
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), before);
}

TEST_F(CIRPipeliningTest, PackCopyUnpack_TwoChunks_Wavefront) {
  auto dev0 = MakePipeDevice(n0, 0);
  auto dev1 = MakePipeDevice(n1, 0);

  // 4 pieces × 128 elements = 512 total.  Chunk size = 256 elements.
  // → 2 chunks of 2 pieces each.
  auto program = MakePackCopyUnpackProgram(dev0, dev1, dt, {128, 128, 128, 128});

  // chunk_size_bytes = 256 elements × 2 bytes (float16) = 512 bytes
  Pipelining pass(256 * torch::elementSize(dt));
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_NO_THROW(Linearity::Check(result));

  // Count operations by type.
  std::size_t num_packs = 0, num_copies = 0, num_unpacks = 0, num_allocs = 0;
  for (std::uint32_t i = 0; i < result.NumOperations(); ++i) {
    auto type = result.Operations()[i].Type();
    if (type == setu::planner::ir::cir::OpType::kPack) num_packs++;
    if (type == setu::planner::ir::cir::OpType::kCopy) num_copies++;
    if (type == setu::planner::ir::cir::OpType::kUnpack) num_unpacks++;
    if (type == setu::planner::ir::cir::OpType::kAllocTmp) num_allocs++;
  }

  // 2 chunks → 2 packs, 2 copies, 2 unpacks, 4 alloc_tmps (2 src + 2 dst).
  EXPECT_EQ(num_packs, 2);
  EXPECT_EQ(num_copies, 2);
  EXPECT_EQ(num_unpacks, 2);
  EXPECT_EQ(num_allocs, 4);
}

TEST_F(CIRPipeliningTest, PackCopyUnpack_CoexistsWithCopyChain) {
  auto dev_a = MakePipeDevice(n0, 0);
  auto dev_b = MakePipeDevice(n0, 1);
  auto dev_c = MakePipeDevice(n1, 0);

  // Build a program with both: a 2-hop CopyChain AND a PackCopyUnpack.
  Program p;
  auto shard = MakePipeEmptyShardRef();

  // --- 2-hop relay chain: dev_a → dev_b → dev_c, 512 elements ---
  auto relay_src = p.EmitView(dev_a, shard, Slice{0, 512}, dt);
  auto relay_dst = p.EmitView(dev_c, shard, Slice{0, 512}, dt);
  auto relay_tmp = p.EmitAllocTmp(dev_b, 512 * torch::elementSize(dt), dt);
  auto rs = p.EmitSlice(relay_src, Slice{0, 512});
  auto rd = p.EmitSlice(relay_dst, Slice{0, 512});
  auto rt = p.EmitSlice(relay_tmp, Slice{0, 512});
  auto c0 = p.EmitCopy(rs, rt);
  (void)p.EmitCopy(c0, rd);
  (void)p.EmitConsume(relay_dst);

  // --- PackCopyUnpack: dev_a → dev_c, 4 × 128 elements ---
  std::vector<Value> pack_srcs, unpack_dsts;
  for (std::size_t i = 0; i < 4; ++i) {
    pack_srcs.push_back(p.EmitView(dev_a, shard, Slice{0, 128}, dt));
    unpack_dsts.push_back(p.EmitView(dev_c, shard, Slice{0, 128}, dt));
  }
  auto pcu_src_tmp = p.EmitAllocTmp(dev_a, 512, dt);
  auto packed = p.EmitPack(pack_srcs, pcu_src_tmp);
  auto pcu_dst_tmp = p.EmitAllocTmp(dev_c, 512, dt);
  auto copied = p.EmitCopy(packed, pcu_dst_tmp);
  (void)p.EmitUnpack(copied, unpack_dsts);

  // chunk_size = 256 elements → both should be pipelined into 2 chunks.
  Pipelining pass(256 * torch::elementSize(dt));
  auto result = pass.Run(std::move(p), DefaultCtx());

  EXPECT_NO_THROW(Linearity::Check(result));

  // Count copies: 2-hop chain with 2 chunks = 4 copies,
  // PCU chain with 2 chunks = 2 copies.  Total = 6.
  std::size_t num_copies = 0;
  for (std::uint32_t i = 0; i < result.NumOperations(); ++i) {
    if (result.Operations()[i].Type() == setu::planner::ir::cir::OpType::kCopy)
      num_copies++;
  }
  EXPECT_EQ(num_copies, 6);
}

TEST_F(CIRPipeliningTest, PackCopyUnpack_Idempotent) {
  auto dev0 = MakePipeDevice(n0, 0);
  auto dev1 = MakePipeDevice(n1, 0);

  auto program = MakePackCopyUnpackProgram(dev0, dev1, dt, {128, 128, 128, 128});

  Pipelining pass(256 * torch::elementSize(dt));
  auto result1 = pass.Run(std::move(program), DefaultCtx());
  auto dump1 = result1.Dump();

  auto result2 = pass.Run(std::move(result1), DefaultCtx());
  auto dump2 = result2.Dump();

  EXPECT_EQ(dump1, dump2);
  EXPECT_NO_THROW(Linearity::Check(result2));
}

}  // namespace
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
