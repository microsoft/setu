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
#include "planner/RegisterSet.h"
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
#include "planner/passes/PassContext.h"
#include "planner/passes/Pipelining.h"
#include "planner/passes/RegisterTiling.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::hints::HintStore;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::Value;
using setu::planner::passes::P2PAccessMap;
using setu::planner::passes::PassContext;
using setu::planner::passes::Pipelining;
using setu::planner::passes::RegisterTiling;
//==============================================================================
namespace {
//==============================================================================

Device MakeTestDevice(std::int16_t gpu_index = 0) {
  auto node_id = boost::uuids::nil_uuid();
  return Device(node_id, setu::commons::datatypes::Device(torch::Device(
                             torch::kCUDA, static_cast<int8_t>(gpu_index))));
}

setu::planner::ir::ref::ShardRef MakeTestShardRef() {
  return setu::planner::ir::ref::ShardRef(boost::uuids::nil_uuid());
}

//==============================================================================

class RegisterTilingTest : public ::testing::Test {
 protected:
  Device dev0 = MakeTestDevice(0);
  Device dev1 = MakeTestDevice(1);
  Device dev2 = MakeTestDevice(2);
  Device dev3 = MakeTestDevice(3);
  torch::Dtype dt = torch::kFloat16;
  setu::planner::ir::ref::ShardRef shard = MakeTestShardRef();
  HintStore hints;
  std::unordered_map<Device, setu::planner::RegisterSet> empty_register_sets;
  P2PAccessMap empty_p2p_access;

  PassContext DefaultCtx() {
    return PassContext{.hints = hints,
                       .register_sets = empty_register_sets,
                       .p2p_access = empty_p2p_access};
  }

  // 128 bytes chunk → 64 float16 elements
  static constexpr std::size_t kChunkBytes = 128;
  static constexpr std::size_t kChunkElements = 64;  // 128 / 2
};

//==============================================================================
// No-op cases
//==============================================================================

TEST_F(RegisterTilingTest, EmptyProgram_ReturnsEmpty) {
  RegisterTiling pass(kChunkBytes);
  Program program;
  auto result = pass.Run(std::move(program), DefaultCtx());
  EXPECT_EQ(result.NumOperations(), 0u);
}

TEST_F(RegisterTilingTest, NoAllocTmp_PassedThrough) {
  RegisterTiling pass(kChunkBytes);
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, 1024}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, 1024}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [2] %2 = copy(%0, %1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(RegisterTilingTest, SmallTmp_PassedThrough) {
  RegisterTiling pass(kChunkBytes);
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, kChunkElements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, kChunkElements}, dt);
  auto tmp = program.EmitAllocTmp(dev2, kChunkElements, dt);
  auto tmp_out = program.EmitCopy(src, tmp);
  (void)program.EmitCopy(tmp_out, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [3] %3 = copy(%0, %2)
  [4] %4 = copy(%3, %1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Single intermediate, large buffer
//==============================================================================

TEST_F(RegisterTilingTest, SingleTmp_LargeBuffer_SplitsIntoChunks) {
  RegisterTiling pass(kChunkBytes);

  // 4 chunks worth of elements
  const std::size_t total = kChunkElements * 4;

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, total}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, total}, dt);
  auto tmp = program.EmitAllocTmp(dev2, total, dt);
  auto tmp_out = program.EmitCopy(src, tmp);
  (void)program.EmitCopy(tmp_out, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 256], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 256], Half)
  [2] %2 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [3] %3 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [5] %5 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [6] %6 = slice(%0, [0, 64])
  [7] %7 = copy(%6, %2)
  [8] %8 = slice(%0, [64, 64])
  [9] %9 = copy(%8, %3)
  [10] %10 = slice(%0, [128, 64])
  [11] %11 = copy(%10, %4)
  [12] %12 = slice(%0, [192, 64])
  [13] %13 = copy(%12, %5)
  [14] %14 = slice(%1, [0, 64])
  [15] %15 = copy(%7, %14)
  [16] %16 = slice(%1, [64, 64])
  [17] %17 = copy(%9, %16)
  [18] %18 = slice(%1, [128, 64])
  [19] %19 = copy(%11, %18)
  [20] %20 = slice(%1, [192, 64])
  [21] %21 = copy(%13, %20)
  [22] %22 = consume(%1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Partial last chunk
//==============================================================================

TEST_F(RegisterTilingTest, PartialLastChunk_CorrectSize) {
  RegisterTiling pass(kChunkBytes);

  // 2.5 chunks worth → 3 AllocTmps, last one has 32 elements
  const std::size_t total = kChunkElements * 2 + kChunkElements / 2;

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, total}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, total}, dt);
  auto tmp = program.EmitAllocTmp(dev2, total, dt);
  auto tmp_out = program.EmitCopy(src, tmp);
  (void)program.EmitCopy(tmp_out, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 160], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 160], Half)
  [2] %2 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [3] %3 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 32, Half)
  [5] %5 = slice(%0, [0, 64])
  [6] %6 = copy(%5, %2)
  [7] %7 = slice(%0, [64, 64])
  [8] %8 = copy(%7, %3)
  [9] %9 = slice(%0, [128, 32])
  [10] %10 = copy(%9, %4)
  [11] %11 = slice(%1, [0, 64])
  [12] %12 = copy(%6, %11)
  [13] %13 = slice(%1, [64, 64])
  [14] %14 = copy(%8, %13)
  [15] %15 = slice(%1, [128, 32])
  [16] %16 = copy(%10, %15)
  [17] %17 = consume(%1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Two intermediates (3-hop chain: A → C → D → B)
//==============================================================================

TEST_F(RegisterTilingTest, TwoIntermediates_BothChunked) {
  RegisterTiling pass(kChunkBytes);

  const std::size_t total = kChunkElements * 3;

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, total}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, total}, dt);
  auto tmp_c = program.EmitAllocTmp(dev2, total, dt);
  auto tmp_d = program.EmitAllocTmp(dev3, total, dt);
  auto tmp_c_out = program.EmitCopy(src, tmp_c);
  auto tmp_d_out = program.EmitCopy(tmp_c_out, tmp_d);
  (void)program.EmitCopy(tmp_d_out, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 192], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 192], Half)
  [2] %2 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [3] %3 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [5] %5 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 64, Half)
  [6] %6 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 64, Half)
  [7] %7 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 64, Half)
  [8] %8 = slice(%0, [0, 64])
  [9] %9 = copy(%8, %2)
  [10] %10 = slice(%0, [64, 64])
  [11] %11 = copy(%10, %3)
  [12] %12 = slice(%0, [128, 64])
  [13] %13 = copy(%12, %4)
  [14] %14 = copy(%9, %5)
  [15] %15 = copy(%11, %6)
  [16] %16 = copy(%13, %7)
  [17] %17 = slice(%1, [0, 64])
  [18] %18 = copy(%14, %17)
  [19] %19 = slice(%1, [64, 64])
  [20] %20 = copy(%15, %19)
  [21] %21 = slice(%1, [128, 64])
  [22] %22 = copy(%16, %21)
  [23] %23 = consume(%1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Direct copy (no tmps) unchanged
//==============================================================================

TEST_F(RegisterTilingTest, DirectCopy_Unchanged) {
  RegisterTiling pass(kChunkBytes);

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, 1024}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, 1024}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [2] %2 = copy(%0, %1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Multiple independent chains
//==============================================================================

TEST_F(RegisterTilingTest, MultipleChains_IndependentlyTiled) {
  RegisterTiling pass(kChunkBytes);

  const std::size_t total = kChunkElements * 2;

  Program program;
  // Chain 1: dev0 → dev2 → dev1
  auto src1 = program.EmitView(dev0, shard, Slice{0, total}, dt);
  auto dst_view = program.EmitView(dev1, shard, Slice{0, total * 2}, dt);
  auto dst1_slice = program.EmitSlice(dst_view, Slice{0, total});
  auto tmp1 = program.EmitAllocTmp(dev2, total, dt);
  auto tmp1_out = program.EmitCopy(src1, tmp1);
  (void)program.EmitCopy(tmp1_out, dst1_slice);

  // Chain 2: dev0 → dev3 → dev1
  auto src2 = program.EmitView(dev0, shard, Slice{total, total}, dt);
  auto dst2_slice = program.EmitSlice(dst_view, Slice{total, total});
  auto tmp2 = program.EmitAllocTmp(dev3, total, dt);
  auto tmp2_out = program.EmitCopy(src2, tmp2);
  (void)program.EmitCopy(tmp2_out, dst2_slice);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 128], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 256], Half)
  [2] %2 = slice(%1, [0, 128])
  [3] %3 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [5] %5 = slice(%0, [0, 64])
  [6] %6 = copy(%5, %3)
  [7] %7 = slice(%0, [64, 64])
  [8] %8 = copy(%7, %4)
  [9] %9 = slice(%2, [0, 64])
  [10] %10 = copy(%6, %9)
  [11] %11 = slice(%2, [64, 64])
  [12] %12 = copy(%8, %11)
  [13] %13 = consume(%2)
  [14] %14 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [128, 128], Half)
  [15] %15 = slice(%1, [128, 128])
  [16] %16 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 64, Half)
  [17] %17 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 64, Half)
  [18] %18 = slice(%14, [0, 64])
  [19] %19 = copy(%18, %16)
  [20] %20 = slice(%14, [64, 64])
  [21] %21 = copy(%20, %17)
  [22] %22 = slice(%15, [0, 64])
  [23] %23 = copy(%19, %22)
  [24] %24 = slice(%15, [64, 64])
  [25] %25 = copy(%21, %24)
  [26] %26 = consume(%15)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// SliceOp on chunked AllocTmpOp
//==============================================================================

TEST_F(RegisterTilingTest, SliceOnChunkedAllocTmp_Aligned) {
  RegisterTiling pass(kChunkBytes);

  // alloc_tmp with 128 elements (2 register chunks of 64)
  // slice first chunk [0, 64], copy through it
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, kChunkElements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, kChunkElements}, dt);
  auto tmp = program.EmitAllocTmp(dev2, kChunkElements * 2, dt);
  auto tmp_slice = program.EmitSlice(tmp, Slice{0, kChunkElements});
  auto tmp_out = program.EmitCopy(src, tmp_slice);
  (void)program.EmitCopy(tmp_out, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [3] %3 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [4] %4 = slice(%0, [0, 64])
  [5] %5 = copy(%4, %2)
  [6] %6 = slice(%1, [0, 64])
  [7] %7 = copy(%5, %6)
  [8] %8 = consume(%1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(RegisterTilingTest, SliceOnChunkedAllocTmp_Unaligned) {
  RegisterTiling pass(kChunkBytes);

  // alloc_tmp with 192 elements (3 register chunks of 64)
  // slice [32, 64] straddles chunks 0 and 1
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, kChunkElements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, kChunkElements}, dt);
  auto tmp = program.EmitAllocTmp(dev2, kChunkElements * 3, dt);
  auto tmp_slice = program.EmitSlice(tmp, Slice{32, kChunkElements});
  auto tmp_out = program.EmitCopy(src, tmp_slice);
  (void)program.EmitCopy(tmp_out, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  // The slice [32, 64] overlaps:
  //   chunk 0 [0,64): local [32,64) → 32 elements
  //   chunk 1 [64,128): local [0,32) → 32 elements
  // So 2 sub-chunks of 32 elements each → 2 copies per hop
  std::uint32_t copy_count = 0;
  for (std::uint32_t i = 0; i < result.NumOperations(); ++i) {
    if (result.Operations()[i].Type() ==
        setu::planner::ir::cir::OpType::kCopy) {
      copy_count++;
    }
  }
  EXPECT_EQ(copy_count, 4u);  // 2 copies per hop × 2 hops
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(RegisterTilingTest, ChainedSlicesOnChunkedAllocTmp) {
  RegisterTiling pass(kChunkBytes);

  // alloc_tmp(256) → 4 register chunks
  // slice(alloc, [0, 128]) → first 2 chunks
  // slice(outer, [0, 64]) → first 1 chunk
  // copy through the inner slice
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, kChunkElements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, kChunkElements}, dt);
  auto tmp = program.EmitAllocTmp(dev2, kChunkElements * 4, dt);
  auto outer_slice = program.EmitSlice(tmp, Slice{0, kChunkElements * 2});
  auto inner_slice = program.EmitSlice(outer_slice, Slice{0, kChunkElements});
  auto tmp_out = program.EmitCopy(src, inner_slice);
  (void)program.EmitCopy(tmp_out, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  // Inner slice resolves to exactly 1 register chunk → 1 copy per hop
  std::uint32_t copy_count = 0;
  for (std::uint32_t i = 0; i < result.NumOperations(); ++i) {
    if (result.Operations()[i].Type() ==
        setu::planner::ir::cir::OpType::kCopy) {
      copy_count++;
    }
  }
  EXPECT_EQ(copy_count, 2u);  // 1 copy per hop × 2 hops
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Pipelining → RegisterTiling composition
//==============================================================================

TEST_F(RegisterTilingTest, PipeliningThenRegisterTiling_TwoHop) {
  // Build a 2-hop relay: dev0 → dev2 → dev1
  // Payload = 256 elements, pipeline chunk = 128, register chunk = 64
  const std::size_t payload = kChunkElements * 4;  // 256

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, payload}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, payload}, dt);
  auto tmp = program.EmitAllocTmp(dev2, payload, dt);
  auto s_src = program.EmitSlice(src, Slice{0, payload});
  auto s_dst = program.EmitSlice(dst, Slice{0, payload});
  auto s_tmp = program.EmitSlice(tmp, Slice{0, payload});
  auto c0 = program.EmitCopy(s_src, s_tmp);
  (void)program.EmitCopy(c0, s_dst);
  (void)program.EmitConsume(dst);

  // Pipeline chunk = 128 elements → 2 pipeline chunks
  Pipelining pipe_pass(kChunkElements * 2);
  auto after_pipe = pipe_pass.Run(std::move(program), DefaultCtx());
  EXPECT_NO_THROW(Linearity::Check(after_pipe));

  // Register tile → 64-element register chunks
  RegisterTiling rt_pass(kChunkBytes);
  auto result = rt_pass.Run(std::move(after_pipe), DefaultCtx());
  EXPECT_NO_THROW(Linearity::Check(result));

  // 2 pipeline chunks × 2 hops = 4 logical copies
  // Each pipeline chunk = 2 register chunks → 2 copies per logical copy
  // Total = 4 × 2 = 8 copies
  std::uint32_t copy_count = 0;
  for (std::uint32_t i = 0; i < result.NumOperations(); ++i) {
    if (result.Operations()[i].Type() ==
        setu::planner::ir::cir::OpType::kCopy) {
      copy_count++;
    }
  }
  EXPECT_EQ(copy_count, 8u);
}

TEST_F(RegisterTilingTest, PipeliningThenRegisterTiling_ThreeHop) {
  // Build a 3-hop relay: dev0 → dev2 → dev3 → dev1
  const std::size_t payload = kChunkElements * 4;  // 256

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, payload}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, payload}, dt);
  auto tmp_c = program.EmitAllocTmp(dev2, payload, dt);
  auto tmp_d = program.EmitAllocTmp(dev3, payload, dt);
  auto s_src = program.EmitSlice(src, Slice{0, payload});
  auto s_dst = program.EmitSlice(dst, Slice{0, payload});
  auto s_tmp_c = program.EmitSlice(tmp_c, Slice{0, payload});
  auto s_tmp_d = program.EmitSlice(tmp_d, Slice{0, payload});
  auto c0 = program.EmitCopy(s_src, s_tmp_c);
  auto c1 = program.EmitCopy(c0, s_tmp_d);
  (void)program.EmitCopy(c1, s_dst);
  (void)program.EmitConsume(dst);

  Pipelining pipe_pass(kChunkElements * 2);  // 2 pipeline chunks
  auto after_pipe = pipe_pass.Run(std::move(program), DefaultCtx());
  EXPECT_NO_THROW(Linearity::Check(after_pipe));

  RegisterTiling rt_pass(kChunkBytes);
  auto result = rt_pass.Run(std::move(after_pipe), DefaultCtx());
  EXPECT_NO_THROW(Linearity::Check(result));

  // 2 pipeline chunks × 3 hops = 6 logical copies
  // Each pipeline chunk = 2 register chunks → 2 copies per logical copy
  // Total = 6 × 2 = 12 copies
  std::uint32_t copy_count = 0;
  for (std::uint32_t i = 0; i < result.NumOperations(); ++i) {
    if (result.Operations()[i].Type() ==
        setu::planner::ir::cir::OpType::kCopy) {
      copy_count++;
    }
  }
  EXPECT_EQ(copy_count, 12u);
}

//==============================================================================
// PackUnpackCopies → RegisterTiling composition
//==============================================================================

TEST_F(RegisterTilingTest, PackUnpackThenRegisterTiling_SourcesStraddleChunks) {
  // 3 cross-device copies GPU0→GPU1 with sizes that don't align to chunks.
  // PackUnpackCopies consolidates them into pack → copy → unpack.
  // RegisterTiling must then chunk through the pack/unpack ops.
  //
  // Sources: 100 + 200 + 50 = 350 elements
  // Chunk size: 64 elements → chunks of [64, 64, 64, 64, 64, 30]

  Program program;
  auto src0 = program.EmitView(dev0, shard, Slice{0, 100}, dt);
  auto src1 = program.EmitView(dev0, shard, Slice{100, 200}, dt);
  auto src2 = program.EmitView(dev0, shard, Slice{300, 50}, dt);
  auto dst0 = program.EmitView(dev1, shard, Slice{0, 100}, dt);
  auto dst1 = program.EmitView(dev1, shard, Slice{100, 200}, dt);
  auto dst2 = program.EmitView(dev1, shard, Slice{300, 50}, dt);

  // Simulate what PackUnpackCopies produces:
  // pack sources into tmp on src device, copy across, unpack into dsts
  auto tmp_src = program.EmitAllocTmp(dev0, 350, dt);
  auto packed = program.EmitPack({src0, src1, src2}, tmp_src);
  auto tmp_dst = program.EmitAllocTmp(dev1, 350, dt);
  auto copied = program.EmitCopy(packed, tmp_dst);
  auto unpacked = program.EmitUnpack(copied, {dst0, dst1, dst2});
  (void)unpacked;

  EXPECT_NO_THROW(Linearity::Check(program));

  RegisterTiling pass(kChunkBytes);
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_NO_THROW(Linearity::Check(result));

  // No alloc_tmp should exceed the chunk size.
  for (const auto& op : result.Operations()) {
    if (op.Type() == setu::planner::ir::cir::OpType::kAllocTmp) {
      auto& alloc = std::get<setu::planner::ir::cir::AllocTmpOp>(op.op);
      EXPECT_LE(alloc.size_elements, kChunkElements);
    }
  }
}

TEST_F(RegisterTilingTest, PackUnpackThenRegisterTiling_AlignedSources) {
  // Sources perfectly align with chunk boundaries → no slicing needed.
  // 2 sources of 64 elements each = 128 total = 2 chunks exactly.

  Program program;
  auto src0 = program.EmitView(dev0, shard, Slice{0, kChunkElements}, dt);
  auto src1 =
      program.EmitView(dev0, shard, Slice{kChunkElements, kChunkElements}, dt);
  auto dst0 = program.EmitView(dev1, shard, Slice{0, kChunkElements}, dt);
  auto dst1 =
      program.EmitView(dev1, shard, Slice{kChunkElements, kChunkElements}, dt);

  auto tmp_src = program.EmitAllocTmp(dev0, kChunkElements * 2, dt);
  auto packed = program.EmitPack({src0, src1}, tmp_src);
  auto tmp_dst = program.EmitAllocTmp(dev1, kChunkElements * 2, dt);
  auto copied = program.EmitCopy(packed, tmp_dst);
  auto unpacked = program.EmitUnpack(copied, {dst0, dst1});
  (void)unpacked;

  EXPECT_NO_THROW(Linearity::Check(program));

  RegisterTiling pass(kChunkBytes);
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_NO_THROW(Linearity::Check(result));

  // Each pack chunk has exactly one source, each unpack dest gets exactly
  // one chunk → all packs have 1 source, all unpacks become copies.
  // Should have 4 copies: 2 pack-into-chunk + 2 cross-device copies...
  // actually the pack(single_src, chunk) is still a pack, and unpack with
  // 1 piece becomes a copy. Let's just verify linearity and no large tmps.
  for (const auto& op : result.Operations()) {
    if (op.Type() == setu::planner::ir::cir::OpType::kAllocTmp) {
      auto& alloc = std::get<setu::planner::ir::cir::AllocTmpOp>(op.op);
      EXPECT_LE(alloc.size_elements, kChunkElements);
    }
  }
}

TEST_F(RegisterTilingTest, PackUnpackThenRegisterTiling_SmallTmpsUnchanged) {
  // Tmp buffers already fit in a single chunk → pass is a no-op.

  Program program;
  auto src0 = program.EmitView(dev0, shard, Slice{0, 30}, dt);
  auto src1 = program.EmitView(dev0, shard, Slice{30, 20}, dt);
  auto dst0 = program.EmitView(dev1, shard, Slice{0, 30}, dt);
  auto dst1 = program.EmitView(dev1, shard, Slice{30, 20}, dt);

  auto tmp_src = program.EmitAllocTmp(dev0, 50, dt);
  auto packed = program.EmitPack({src0, src1}, tmp_src);
  auto tmp_dst = program.EmitAllocTmp(dev1, 50, dt);
  auto copied = program.EmitCopy(packed, tmp_dst);
  auto unpacked = program.EmitUnpack(copied, {dst0, dst1});
  (void)unpacked;

  auto original_dump = program.Dump();

  RegisterTiling pass(kChunkBytes);
  auto result = pass.Run(std::move(program), DefaultCtx());

  // Small tmps (50 < 64) → no transformation
  EXPECT_EQ(result.Dump(), original_dump);
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
}  // namespace
}  // namespace setu::test::native
//==============================================================================
