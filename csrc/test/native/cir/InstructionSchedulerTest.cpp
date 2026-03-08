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
#include "planner/passes/InstructionScheduler.h"
#include "planner/passes/PassContext.h"
#include "planner/passes/Pipelining.h"
#include "planner/passes/RegisterTiling.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::RegisterSet;
using setu::planner::hints::HintStore;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::LivenessInfo;
using setu::planner::ir::cir::OpType;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::Value;
using setu::planner::passes::InstructionScheduler;
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
  std::unordered_map<Device, RegisterSet> empty_register_sets;

  PassContext DefaultCtx() {
    return PassContext{.hints = hints, .register_sets = empty_register_sets};
  }

  static constexpr std::size_t kChunkBytes = 128;
  static constexpr std::size_t kChunkElements = 64;
};

//==============================================================================
// No-op cases
//==============================================================================

TEST_F(InstructionSchedulerTest, EmptyProgram_ReturnsEmpty) {
  InstructionScheduler pass;
  Program program;
  auto result = pass.Run(std::move(program), DefaultCtx());
  EXPECT_EQ(result.NumOperations(), 0u);
}

TEST_F(InstructionSchedulerTest, SingleOp_Unchanged) {
  InstructionScheduler pass;
  Program program;
  (void)program.EmitView(dev0, shard, Slice{0, 1024}, dt);
  auto result = pass.Run(std::move(program), DefaultCtx());
  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
)");
}

TEST_F(InstructionSchedulerTest, AlreadyOptimal_Unchanged) {
  InstructionScheduler pass;
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
  auto tiled = tiling.Run(std::move(program), DefaultCtx());
  auto pre_pressure = MaxLiveAllocTmps(tiled);

  // Schedule it.
  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(tiled), DefaultCtx());

  EXPECT_EQ(scheduled.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 256], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 256], Half)
  [2] %2 = slice(%0, [0, 64])
  [3] %3 = slice(%0, [64, 64])
  [4] %4 = slice(%0, [128, 64])
  [5] %5 = slice(%0, [192, 64])
  [6] %6 = slice(%1, [0, 64])
  [7] %7 = slice(%1, [64, 64])
  [8] %8 = slice(%1, [128, 64])
  [9] %9 = slice(%1, [192, 64])
  [10] %10 = consume(%1)
  [11] %11 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [12] %12 = copy(%2, %11)
  [13] %13 = copy(%12, %6)
  [14] %14 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [15] %15 = copy(%3, %14)
  [16] %16 = copy(%15, %7)
  [17] %17 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [18] %18 = copy(%4, %17)
  [19] %19 = copy(%18, %8)
  [20] %20 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [21] %21 = copy(%5, %20)
  [22] %22 = copy(%21, %9)
)");
  EXPECT_NO_THROW(Linearity::Check(scheduled));

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
  auto tiled = tiling.Run(std::move(program), DefaultCtx());

  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(tiled), DefaultCtx());

  EXPECT_EQ(scheduled.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 192], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 192], Half)
  [2] %2 = slice(%0, [0, 64])
  [3] %3 = slice(%0, [64, 64])
  [4] %4 = slice(%0, [128, 64])
  [5] %5 = slice(%1, [0, 64])
  [6] %6 = slice(%1, [64, 64])
  [7] %7 = slice(%1, [128, 64])
  [8] %8 = consume(%1)
  [9] %9 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [10] %10 = copy(%2, %9)
  [11] %11 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [12] %12 = copy(%3, %11)
  [13] %13 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [14] %14 = copy(%4, %13)
  [15] %15 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 64, Half)
  [16] %16 = copy(%10, %15)
  [17] %17 = copy(%16, %5)
  [18] %18 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 64, Half)
  [19] %19 = copy(%12, %18)
  [20] %20 = copy(%19, %6)
  [21] %21 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 64, Half)
  [22] %22 = copy(%14, %21)
  [23] %23 = copy(%22, %7)
)");
  EXPECT_NO_THROW(Linearity::Check(scheduled));

  auto pressure = MaxLiveAllocTmps(scheduled);
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

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [2] %2 = copy(%0, %1)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [1024, 512], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [1024, 512], Half)
  [5] %5 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 512, Half)
  [6] %6 = copy(%3, %5)
  [7] %7 = copy(%6, %4)
)");
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
  auto tiled = tiling.Run(std::move(program), DefaultCtx());

  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(tiled), DefaultCtx());

  EXPECT_EQ(scheduled.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 512], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 512], Half)
  [2] %2 = slice(%0, [0, 64])
  [3] %3 = slice(%0, [64, 64])
  [4] %4 = slice(%0, [128, 64])
  [5] %5 = slice(%0, [192, 64])
  [6] %6 = slice(%0, [256, 64])
  [7] %7 = slice(%0, [320, 64])
  [8] %8 = slice(%0, [384, 64])
  [9] %9 = slice(%0, [448, 64])
  [10] %10 = slice(%1, [0, 64])
  [11] %11 = slice(%1, [64, 64])
  [12] %12 = slice(%1, [128, 64])
  [13] %13 = slice(%1, [192, 64])
  [14] %14 = slice(%1, [256, 64])
  [15] %15 = slice(%1, [320, 64])
  [16] %16 = slice(%1, [384, 64])
  [17] %17 = slice(%1, [448, 64])
  [18] %18 = consume(%1)
  [19] %19 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [20] %20 = copy(%2, %19)
  [21] %21 = copy(%20, %10)
  [22] %22 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [23] %23 = copy(%3, %22)
  [24] %24 = copy(%23, %11)
  [25] %25 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [26] %26 = copy(%4, %25)
  [27] %27 = copy(%26, %12)
  [28] %28 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [29] %29 = copy(%5, %28)
  [30] %30 = copy(%29, %13)
  [31] %31 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [32] %32 = copy(%6, %31)
  [33] %33 = copy(%32, %14)
  [34] %34 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [35] %35 = copy(%7, %34)
  [36] %36 = copy(%35, %15)
  [37] %37 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [38] %38 = copy(%8, %37)
  [39] %39 = copy(%38, %16)
  [40] %40 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [41] %41 = copy(%9, %40)
  [42] %42 = copy(%41, %17)
)");
  EXPECT_NO_THROW(Linearity::Check(scheduled));
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
  auto scheduled = scheduler.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(scheduled.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 256], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 256], Half)
  [2] %2 = slice(%0, [0, 64])
  [3] %3 = slice(%0, [64, 64])
  [4] %4 = slice(%0, [128, 64])
  [5] %5 = slice(%0, [192, 64])
  [6] %6 = slice(%1, [0, 64])
  [7] %7 = slice(%1, [64, 64])
  [8] %8 = slice(%1, [128, 64])
  [9] %9 = slice(%1, [192, 64])
  [10] %10 = consume(%1)
  [11] %11 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [12] %12 = copy(%2, %11)
  [13] %13 = copy(%12, %6)
  [14] %14 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [15] %15 = copy(%3, %14)
  [16] %16 = copy(%15, %7)
  [17] %17 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [18] %18 = copy(%4, %17)
  [19] %19 = copy(%18, %8)
  [20] %20 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [21] %21 = copy(%5, %20)
  [22] %22 = copy(%21, %9)
)");
  EXPECT_NO_THROW(Linearity::Check(scheduled));

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
  auto scheduled = scheduler.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(scheduled.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 64], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 64], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [128, 64], Half)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [128, 64], Half)
  [6] %6 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [192, 64], Half)
  [7] %7 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [192, 64], Half)
  [8] %8 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [256, 64], Half)
  [9] %9 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [256, 64], Half)
  [10] %10 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [320, 64], Half)
  [11] %11 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [320, 64], Half)
  [12] %12 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [13] %13 = copy(%0, %12)
  [14] %14 = copy(%13, %1)
  [15] %15 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [16] %16 = copy(%2, %15)
  [17] %17 = copy(%16, %3)
  [18] %18 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [19] %19 = copy(%4, %18)
  [20] %20 = copy(%19, %5)
  [21] %21 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [22] %22 = copy(%6, %21)
  [23] %23 = copy(%22, %7)
  [24] %24 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [25] %25 = copy(%8, %24)
  [26] %26 = copy(%25, %9)
  [27] %27 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [28] %28 = copy(%10, %27)
  [29] %29 = copy(%28, %11)
)");
  EXPECT_NO_THROW(Linearity::Check(scheduled));

  auto pressure = MaxLiveAllocTmps(scheduled);
  EXPECT_LE(pressure, 2u) << "Independent chains: expected pressure <= 2, got "
                          << pressure;
}

//==============================================================================
// Pressure guard: respects register limit
//==============================================================================

TEST_F(InstructionSchedulerTest, PressureGuard_RespectsRegisterLimit) {
  // 4 independent AllocTmpOps on the same device, but only 2 registers.
  // The pressure guard must defer AllocTmpOps so that at most 2 are live
  // at any point.
  const std::size_t n = kChunkElements;

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, n * 4}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, n * 4}, dt);

  auto t0 = program.EmitAllocTmp(dev2, n, dt);
  auto t1 = program.EmitAllocTmp(dev2, n, dt);
  auto t2 = program.EmitAllocTmp(dev2, n, dt);
  auto t3 = program.EmitAllocTmp(dev2, n, dt);

  auto s0 = program.EmitSlice(src, Slice{0 * n, n});
  auto t0_out = program.EmitCopy(s0, t0);
  auto s1 = program.EmitSlice(src, Slice{1 * n, n});
  auto t1_out = program.EmitCopy(s1, t1);
  auto s2 = program.EmitSlice(src, Slice{2 * n, n});
  auto t2_out = program.EmitCopy(s2, t2);
  auto s3 = program.EmitSlice(src, Slice{3 * n, n});
  auto t3_out = program.EmitCopy(s3, t3);

  auto d0 = program.EmitSlice(dst, Slice{0 * n, n});
  (void)program.EmitCopy(t0_out, d0);
  auto d1 = program.EmitSlice(dst, Slice{1 * n, n});
  (void)program.EmitCopy(t1_out, d1);
  auto d2 = program.EmitSlice(dst, Slice{2 * n, n});
  (void)program.EmitCopy(t2_out, d2);
  auto d3 = program.EmitSlice(dst, Slice{3 * n, n});
  (void)program.EmitCopy(t3_out, d3);

  (void)program.EmitConsume(dst);

  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev2, RegisterSet::Uniform(2, 1024)}};
  PassContext ctx{.hints = hints, .register_sets = register_sets};
  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(program), ctx);

  EXPECT_NO_THROW(Linearity::Check(scheduled));

  auto pressure = MaxLiveAllocTmps(scheduled);
  EXPECT_LE(pressure, 2u)
      << "Pressure guard should limit live AllocTmps to 2, got " << pressure;
}

//==============================================================================
// Pressure guard: multi-hop relay bounded by pool
//==============================================================================

TEST_F(InstructionSchedulerTest, PressureGuard_MultiHop_BoundedByPool) {
  // Multi-hop relay A → C → D → B, tiled into 3 chunks.
  // Each relay device (C, D) has only 2 registers.
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
  auto tiled = tiling.Run(std::move(program), DefaultCtx());

  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev2, RegisterSet::Uniform(2, 1024)},
      {dev3, RegisterSet::Uniform(2, 1024)}};
  PassContext ctx{.hints = hints, .register_sets = register_sets};
  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(tiled), ctx);

  EXPECT_NO_THROW(Linearity::Check(scheduled));

  auto pressure = MaxLiveAllocTmps(scheduled);
  EXPECT_LE(pressure, 4u)
      << "Multi-hop pressure should be bounded by register pools, got "
      << pressure;
}

//==============================================================================
// Wavefront preservation: Pipelining → InstructionScheduler should not
// reorder when pressure already fits.
//==============================================================================

TEST_F(InstructionSchedulerTest, PipelinedProgram_PreservesWavefrontOrder) {
  // 2-hop relay: A(dev0) → C(dev2) → B(dev1), payload = 128 elements,
  // pipeline chunk = 64 elements → 2 pipeline chunks.
  const std::size_t total = kChunkElements * 2;

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, total}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, total}, dt);
  auto tmp = program.EmitAllocTmp(dev2, total, dt);
  auto tmp_out = program.EmitCopy(src, tmp);
  (void)program.EmitCopy(tmp_out, dst);

  // Run Pipelining with chunk_size = 64 elements.
  Pipelining pipelining(kChunkElements);
  auto pipelined = pipelining.Run(std::move(program), DefaultCtx());
  auto pipelined_dump = pipelined.Dump();

  // Run InstructionScheduler with enough registers (budget fits).
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev2, RegisterSet::Uniform(2, 1024)}};
  PassContext ctx{.hints = hints, .register_sets = register_sets};
  InstructionScheduler scheduler;
  auto scheduled = scheduler.Run(std::move(pipelined), ctx);

  EXPECT_NO_THROW(Linearity::Check(scheduled));

  // The scheduler should NOT have reordered — output should match
  // Pipelining's wavefront order exactly.
  EXPECT_EQ(scheduled.Dump(), pipelined_dump)
      << "InstructionScheduler should preserve wavefront order when "
         "pressure fits within register budget";
}

//==============================================================================
}  // namespace
}  // namespace setu::test::native
//==============================================================================
