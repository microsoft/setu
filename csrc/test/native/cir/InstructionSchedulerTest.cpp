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
#include "planner/passes/InstructionScheduler.h"
#include "planner/passes/RegisterTiling.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::hints::HintStore;
using setu::planner::ir::cir::AllocTmpOp;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::LivenessInfo;
using setu::planner::ir::cir::OpType;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::Value;
using setu::planner::passes::InstructionScheduler;
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

/// Compute the maximum number of simultaneously live AllocTmp values at any
/// point in the program.
std::uint32_t MaxLiveAllocTmps(const Program& program) {
  auto liveness = LivenessInfo::Build(program);

  // Collect which values come from AllocTmpOps.
  std::unordered_set<std::uint32_t> alloc_tmp_values;
  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    if (op.Type() == OpType::kAllocTmp) {
      for (const auto& def : op.Defs()) {
        alloc_tmp_values.insert(def.id);
      }
    }
  }

  std::uint32_t max_live = 0;
  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    auto live = liveness.LiveAt(op_idx);
    std::uint32_t count = 0;
    for (const auto& v : live) {
      if (alloc_tmp_values.contains(v.id)) {
        ++count;
      }
    }
    max_live = std::max(max_live, count);
  }
  return max_live;
}

//==============================================================================

class InstructionSchedulerTest : public ::testing::Test {
 protected:
  Device dev0 = MakeTestDevice(0);
  Device dev1 = MakeTestDevice(1);
  Device dev2 = MakeTestDevice(2);
  Device dev3 = MakeTestDevice(3);
  torch::Dtype dt = torch::kFloat16;
  setu::planner::ir::ref::ShardRef shard = MakeTestShardRef();
  HintStore hints;

  static constexpr std::size_t kChunkBytes = 128;
  static constexpr std::size_t kChunkElements = 64;
};

//==============================================================================
// No-op cases
//==============================================================================

TEST_F(InstructionSchedulerTest, EmptyProgram_ReturnsEmpty) {
  InstructionScheduler pass;
  Program program;
  auto result = pass.Run(std::move(program), hints);
  EXPECT_EQ(result.NumOperations(), 0u);
}

TEST_F(InstructionSchedulerTest, SingleOp_Unchanged) {
  InstructionScheduler pass;
  Program program;
  (void)program.EmitView(dev0, shard, Slice{0, 1024}, dt);
  auto result = pass.Run(std::move(program), hints);
  EXPECT_EQ(result.NumOperations(), 1u);
}

TEST_F(InstructionSchedulerTest, AlreadyOptimal_Unchanged) {
  InstructionScheduler pass;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, 1024}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, 1024}, dt);
  (void)program.EmitCopy(src, dst);

  auto num_ops = program.NumOperations();
  auto result = pass.Run(std::move(program), hints);
  EXPECT_EQ(result.NumOperations(), num_ops);
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Flat tiled output: scheduler should interleave alloc/write/read
//==============================================================================

TEST_F(InstructionSchedulerTest, FlatTiledOutput_ReducesRegisterPressure) {
  // Build a program as RegisterTiling would produce: flat allocs then copies.
  const std::size_t total = kChunkElements * 4;

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, total}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, total}, dt);
  auto tmp = program.EmitAllocTmp(dev2, total, dt);
  auto tmp_out = program.EmitCopy(src, tmp);
  (void)program.EmitCopy(tmp_out, dst);

  // Tile it first.
  RegisterTiling tiling(kChunkBytes);
  auto tiled = tiling.Run(std::move(program), hints);
  auto pre_pressure = MaxLiveAllocTmps(tiled);

  // Schedule it.
  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(tiled), hints);

  EXPECT_NO_THROW(Linearity::Check(scheduled));
  EXPECT_EQ(CountOps(scheduled, OpType::kAllocTmp), 4u);
  EXPECT_EQ(CountOps(scheduled, OpType::kCopy), 8u);

  auto post_pressure = MaxLiveAllocTmps(scheduled);
  EXPECT_LE(post_pressure, 2u)
      << "Scheduler should reduce pressure from " << pre_pressure << " to <=2";
}

//==============================================================================
// Multi-hop: A → C → D → B, tiled then scheduled
//==============================================================================

TEST_F(InstructionSchedulerTest, MultiHop_TiledAndScheduled_LowPressure) {
  const std::size_t total = kChunkElements * 3;

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, total}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, total}, dt);
  auto tmp_c = program.EmitAllocTmp(dev2, total, dt);
  auto tmp_d = program.EmitAllocTmp(dev3, total, dt);
  auto tmp_c_out = program.EmitCopy(src, tmp_c);
  auto tmp_d_out = program.EmitCopy(tmp_c_out, tmp_d);
  (void)program.EmitCopy(tmp_d_out, dst);

  RegisterTiling tiling(kChunkBytes);
  auto tiled = tiling.Run(std::move(program), hints);

  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(tiled), hints);

  EXPECT_NO_THROW(Linearity::Check(scheduled));
  // 3 chunks × 2 intermediates = 6 AllocTmps
  EXPECT_EQ(CountOps(scheduled, OpType::kAllocTmp), 6u);

  auto pressure = MaxLiveAllocTmps(scheduled);
  // Should need at most ~2 per device (current + next), so ≤4 total
  EXPECT_LE(pressure, 4u) << "Multi-hop pressure should be bounded";
}

//==============================================================================
// Mixed program: tmps + non-tmp ops
//==============================================================================

TEST_F(InstructionSchedulerTest, MixedProgram_PreservesCorrectness) {
  InstructionScheduler pass;

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, 1024}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, 1024}, dt);
  // A direct copy (no tmp)
  (void)program.EmitCopy(src, dst);
  // A separate tmp-mediated copy
  auto src2 = program.EmitView(dev0, shard, Slice{1024, 512}, dt);
  auto dst2 = program.EmitView(dev1, shard, Slice{1024, 512}, dt);
  auto tmp = program.EmitAllocTmp(dev2, 512, dt);
  auto tmp_out = program.EmitCopy(src2, tmp);
  (void)program.EmitCopy(tmp_out, dst2);

  auto num_ops = program.NumOperations();
  auto result = pass.Run(std::move(program), hints);

  EXPECT_EQ(result.NumOperations(), num_ops);
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Linearity preserved after scheduling
//==============================================================================

TEST_F(InstructionSchedulerTest, ComplexProgram_LinearityPreserved) {
  const std::size_t total = kChunkElements * 8;

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, total}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, total}, dt);
  auto tmp = program.EmitAllocTmp(dev2, total, dt);
  auto tmp_out = program.EmitCopy(src, tmp);
  (void)program.EmitCopy(tmp_out, dst);

  RegisterTiling tiling(kChunkBytes);
  auto tiled = tiling.Run(std::move(program), hints);

  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(tiled), hints);

  EXPECT_NO_THROW(Linearity::Check(scheduled));
  EXPECT_EQ(CountOps(scheduled, OpType::kAllocTmp), 8u);
}

//==============================================================================
// Copy-chain values: tmp-origin values flowing through copies are not direct
// AllocTmpOp outputs but still represent live registers.  The scheduler must
// bound pressure for these too.
//==============================================================================

TEST_F(InstructionSchedulerTest, CopyChainValues_PressureBounded) {
  // Manually construct the flat pattern RegisterTiling would produce for a
  // 4-chunk single-hop transfer: A(dev0) → tmp(dev2) → B(dev1).
  //
  // Flat order (worst case):
  //   4× AllocTmp, 4× (Slice+Copy into tmp), 4× (Slice+Copy out of tmp)
  //
  // After the write copies, c0_out..c3_out are all live simultaneously.
  // These are CopyOp outputs, NOT AllocTmpOp outputs.  The scheduler must
  // still interleave to avoid this pile-up.
  const std::size_t n = kChunkElements;

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, n * 4}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, n * 4}, dt);

  // Flat AllocTmps.
  auto t0 = program.EmitAllocTmp(dev2, n, dt);
  auto t1 = program.EmitAllocTmp(dev2, n, dt);
  auto t2 = program.EmitAllocTmp(dev2, n, dt);
  auto t3 = program.EmitAllocTmp(dev2, n, dt);

  // Flat writes: src slices → tmps.
  auto s0 = program.EmitSlice(src, Slice{0 * n, n});
  auto t0_out = program.EmitCopy(s0, t0);
  auto s1 = program.EmitSlice(src, Slice{1 * n, n});
  auto t1_out = program.EmitCopy(s1, t1);
  auto s2 = program.EmitSlice(src, Slice{2 * n, n});
  auto t2_out = program.EmitCopy(s2, t2);
  auto s3 = program.EmitSlice(src, Slice{3 * n, n});
  auto t3_out = program.EmitCopy(s3, t3);

  // Flat reads: tmps → dst slices.
  auto d0 = program.EmitSlice(dst, Slice{0 * n, n});
  (void)program.EmitCopy(t0_out, d0);
  auto d1 = program.EmitSlice(dst, Slice{1 * n, n});
  (void)program.EmitCopy(t1_out, d1);
  auto d2 = program.EmitSlice(dst, Slice{2 * n, n});
  (void)program.EmitCopy(t2_out, d2);
  auto d3 = program.EmitSlice(dst, Slice{3 * n, n});
  (void)program.EmitCopy(t3_out, d3);

  (void)program.EmitConsume(dst);

  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(program), hints);

  EXPECT_NO_THROW(Linearity::Check(scheduled));
  EXPECT_EQ(CountOps(scheduled, OpType::kAllocTmp), 4u);

  auto pressure = MaxLiveAllocTmps(scheduled);
  EXPECT_LE(pressure, 2u)
      << "Copy-chain values should not cause pressure to exceed 2, got "
      << pressure;
}

//==============================================================================
// Many independent tmp chains: N independent A → tmp → B transfers.
// Without interleaving, all N tmps would be live simultaneously.
//==============================================================================

TEST_F(InstructionSchedulerTest, ManyIndependentChains_PressureBounded) {
  Program program;

  // 6 independent single-chunk transfers, each through its own tmp.
  constexpr std::size_t kNumChains = 6;
  for (std::size_t i = 0; i < kNumChains; ++i) {
    auto src = program.EmitView(dev0, shard, Slice{i * 64, 64}, dt);
    auto dst = program.EmitView(dev1, shard, Slice{i * 64, 64}, dt);
    auto tmp = program.EmitAllocTmp(dev2, 64, dt);
    auto tmp_out = program.EmitCopy(src, tmp);
    (void)program.EmitCopy(tmp_out, dst);
  }

  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(program), hints);

  EXPECT_NO_THROW(Linearity::Check(scheduled));
  EXPECT_EQ(CountOps(scheduled, OpType::kAllocTmp), kNumChains);

  auto pressure = MaxLiveAllocTmps(scheduled);
  // The scheduler should drain each chain before starting the next,
  // keeping at most ~2 tmps live (current + one being set up).
  EXPECT_LE(pressure, 2u) << "Independent chains: expected pressure <= 2, got "
                          << pressure;
}

//==============================================================================
}  // namespace
}  // namespace setu::test::native
//==============================================================================
