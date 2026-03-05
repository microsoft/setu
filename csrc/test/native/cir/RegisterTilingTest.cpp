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
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
#include "planner/passes/RegisterTiling.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::hints::HintStore;
using setu::planner::ir::cir::AllocTmpOp;
using setu::planner::ir::cir::CopyOp;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::OpType;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::Value;
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

std::size_t CountOps(const Program& program, OpType type) {
  std::size_t count = 0;
  for (const auto& op : program.Operations()) {
    if (op.Type() == type) {
      ++count;
    }
  }
  return count;
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
  auto result = pass.Run(std::move(program), hints);
  EXPECT_EQ(result.NumOperations(), 0u);
}

TEST_F(RegisterTilingTest, NoAllocTmp_PassedThrough) {
  RegisterTiling pass(kChunkBytes);
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, 1024}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, 1024}, dt);
  (void)program.EmitCopy(src, dst);

  auto num_ops = program.NumOperations();
  auto result = pass.Run(std::move(program), hints);

  EXPECT_EQ(result.NumOperations(), num_ops);
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

  auto num_ops = program.NumOperations();
  auto result = pass.Run(std::move(program), hints);

  EXPECT_EQ(result.NumOperations(), num_ops);
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

  auto result = pass.Run(std::move(program), hints);

  // 1 large AllocTmp → 4 register-sized AllocTmps
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 4u);
  // 4 write copies + 4 read copies
  EXPECT_EQ(CountOps(result, OpType::kCopy), 8u);
  // 4 src slices + 4 dst slices
  EXPECT_EQ(CountOps(result, OpType::kSlice), 8u);
  // 1 consume for the dst
  EXPECT_EQ(CountOps(result, OpType::kConsume), 1u);
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

  auto result = pass.Run(std::move(program), hints);

  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 3u);

  // Verify last AllocTmp has the partial size
  std::size_t alloc_count = 0;
  for (const auto& op : result.Operations()) {
    if (op.Type() == OpType::kAllocTmp) {
      const auto& alloc = std::get<AllocTmpOp>(op.op);
      if (alloc_count < 2) {
        EXPECT_EQ(alloc.size_elements, kChunkElements);
      } else {
        EXPECT_EQ(alloc.size_elements, kChunkElements / 2);
      }
      ++alloc_count;
    }
  }
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

  auto result = pass.Run(std::move(program), hints);

  // 3 chunks × 2 intermediates = 6 AllocTmps
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 6u);
  // 3 write-to-C + 3 relay C→D + 3 read-from-D = 9 copies
  EXPECT_EQ(CountOps(result, OpType::kCopy), 9u);
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

  auto result = pass.Run(std::move(program), hints);

  EXPECT_EQ(CountOps(result, OpType::kCopy), 1u);
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 0u);
  EXPECT_EQ(CountOps(result, OpType::kSlice), 0u);
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

  auto result = pass.Run(std::move(program), hints);

  // 2 chains × 2 chunks = 4 AllocTmps
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 4u);
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
}  // namespace
}  // namespace setu::test::native
//==============================================================================
