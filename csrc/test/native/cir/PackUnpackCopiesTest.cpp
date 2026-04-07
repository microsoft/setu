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
#include "planner/hints/HintStore.h"
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
#include "planner/passes/PackUnpackCopies.h"
#include "planner/passes/PassContext.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::Participant;
using setu::planner::hints::HintStore;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::passes::PackUnpackCopies;
using setu::planner::passes::P2PAccessMap;
using setu::planner::passes::PassContext;
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

class PackUnpackCopiesTest : public ::testing::Test {
 protected:
  Device dev0 = MakeTestDevice(0);
  Device dev1 = MakeTestDevice(1);
  Device dev2 = MakeTestDevice(2);
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
};

//==============================================================================
// Empty / no-op cases
//==============================================================================

TEST_F(PackUnpackCopiesTest, EmptyProgram_ProducesEmptyProgram) {
  Program program;
  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.NumOperations(), 0u);
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(PackUnpackCopiesTest, ViewsOnly_NoCopies_PassedThrough) {
  // Program with only views and no copies should be unchanged.
  Program program;
  (void)program.EmitView(dev0, shard, Slice{0, 64}, dt);
  (void)program.EmitView(dev1, shard, Slice{0, 32}, dt);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 32], Half)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Same-device copies should never be grouped
//==============================================================================

TEST_F(PackUnpackCopiesTest, SameDeviceCopies_NeverGrouped) {
  // Multiple copies on the same device should remain as individual copies.
  Program program;
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev0, shard, Slice{64, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{128, 32}, dt);
  auto d1 = program.EmitView(dev0, shard, Slice{160, 32}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [128, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [160, 32], Half)
  [4] %4 = copy(%0, %1)
  [5] %5 = copy(%2, %3)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Singleton cross-device copies should not be packed (no benefit)
//==============================================================================

TEST_F(PackUnpackCopiesTest, SingleCrossDeviceCopy_NotPacked) {
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  (void)program.EmitCopy(src, dst);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = copy(%0, %1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Two cross-device copies consolidated into pack → copy → unpack
//==============================================================================

TEST_F(PackUnpackCopiesTest, TwoCrossDeviceCopies_ConsolidatedIntoOne) {
  Program program;
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, dt);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, Half)
  [5] %5 = pack((%0, %2), %4)
  [6] %6 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, Half)
  [7] %7 = copy(%5, %6)
  [8] (%8, %9) = unpack(%7, (%1, %3))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Temp buffer sizes must equal the total size of all packed sources
//==============================================================================

TEST_F(PackUnpackCopiesTest, TempBufferSize_EqualsSum) {
  // Two sources of size 64 and 32 → temp buffer should be 96 elements.
  Program program;
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, dt);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, Half)
  [5] %5 = pack((%0, %2), %4)
  [6] %6 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, Half)
  [7] %7 = copy(%5, %6)
  [8] (%8, %9) = unpack(%7, (%1, %3))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Temp buffers must be allocated on the correct devices
//==============================================================================

TEST_F(PackUnpackCopiesTest, TempBuffers_AllocatedOnCorrectDevices) {
  // Copies go dev0 → dev1, so one temp on dev0 (pack side), one on dev1.
  Program program;
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, dt);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, Half)
  [5] %5 = pack((%0, %2), %4)
  [6] %6 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, Half)
  [7] %7 = copy(%5, %6)
  [8] (%8, %9) = unpack(%7, (%1, %3))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Temp buffers must have the correct dtype
//==============================================================================

TEST_F(PackUnpackCopiesTest, TempBuffers_MatchSourceDtype) {
  auto dtype = torch::kBFloat16;
  Program program;
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dtype);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dtype);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, dtype);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, dtype);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], BFloat16)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], BFloat16)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], BFloat16)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], BFloat16)
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, BFloat16)
  [5] %5 = pack((%0, %2), %4)
  [6] %6 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, BFloat16)
  [7] %7 = copy(%5, %6)
  [8] (%8, %9) = unpack(%7, (%1, %3))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Pack should have correct number of sources, unpack correct destinations
//==============================================================================

TEST_F(PackUnpackCopiesTest, PackSources_MatchNumberOfGroupedCopies) {
  Program program;
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, dt);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, dt);
  auto s2 = program.EmitView(dev0, shard, Slice{96, 16}, dt);
  auto d2 = program.EmitView(dev1, shard, Slice{96, 16}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);
  (void)program.EmitCopy(s2, d2);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [96, 16], Half)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [96, 16], Half)
  [6] %6 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 112, Half)
  [7] %7 = pack((%0, %2, %4), %6)
  [8] %8 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 112, Half)
  [9] %9 = copy(%7, %8)
  [10] (%10, %11, %12) = unpack(%9, (%1, %3, %5))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Different device pairs must form separate groups
//==============================================================================

TEST_F(PackUnpackCopiesTest, DifferentDevicePairs_SeparateGroups) {
  Program program;
  // Two copies dev0 → dev1
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, dt);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  // Two copies dev0 → dev2
  auto s2 = program.EmitView(dev0, shard, Slice{0, 48}, dt);
  auto d2 = program.EmitView(dev2, shard, Slice{0, 48}, dt);
  auto s3 = program.EmitView(dev0, shard, Slice{48, 16}, dt);
  auto d3 = program.EmitView(dev2, shard, Slice{48, 16}, dt);
  (void)program.EmitCopy(s2, d2);
  (void)program.EmitCopy(s3, d3);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 48], Half)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 48], Half)
  [6] %6 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [48, 16], Half)
  [7] %7 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [48, 16], Half)
  [8] %8 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, Half)
  [9] %9 = pack((%0, %2), %8)
  [10] %10 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, Half)
  [11] %11 = copy(%9, %10)
  [12] (%12, %13) = unpack(%11, (%1, %3))
  [13] %14 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 64, Half)
  [14] %15 = pack((%4, %6), %14)
  [15] %16 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [16] %17 = copy(%15, %16)
  [17] (%18, %19) = unpack(%17, (%5, %7))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Opposite direction copies are separate groups (dev0→dev1 vs dev1→dev0)
//==============================================================================

TEST_F(PackUnpackCopiesTest, BidirectionalCopies_SeparateGroups) {
  Program program;
  // Two copies dev0 → dev1
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, dt);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  // Two copies dev1 → dev0 (opposite direction)
  auto s2 = program.EmitView(dev1, shard, Slice{0, 48}, dt);
  auto d2 = program.EmitView(dev0, shard, Slice{0, 48}, dt);
  auto s3 = program.EmitView(dev1, shard, Slice{48, 16}, dt);
  auto d3 = program.EmitView(dev0, shard, Slice{48, 16}, dt);
  (void)program.EmitCopy(s2, d2);
  (void)program.EmitCopy(s3, d3);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 48], Half)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 48], Half)
  [6] %6 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [48, 16], Half)
  [7] %7 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [48, 16], Half)
  [8] %8 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, Half)
  [9] %9 = pack((%0, %2), %8)
  [10] %10 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, Half)
  [11] %11 = copy(%9, %10)
  [12] (%12, %13) = unpack(%11, (%1, %3))
  [13] %14 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 64, Half)
  [14] %15 = pack((%4, %6), %14)
  [15] %16 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 64, Half)
  [16] %17 = copy(%15, %16)
  [17] (%18, %19) = unpack(%17, (%5, %7))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Different dtypes between the same device pair must form separate groups
//==============================================================================

TEST_F(PackUnpackCopiesTest, DifferentDtypes_SeparateGroups) {
  Program program;
  // Two f16 copies (should be grouped)
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, torch::kFloat16);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, torch::kFloat16);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, torch::kFloat16);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, torch::kFloat16);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  // Two f32 copies (should be grouped separately)
  auto s2 = program.EmitView(dev0, shard, Slice{0, 16}, torch::kFloat32);
  auto d2 = program.EmitView(dev1, shard, Slice{0, 16}, torch::kFloat32);
  auto s3 = program.EmitView(dev0, shard, Slice{16, 8}, torch::kFloat32);
  auto d3 = program.EmitView(dev1, shard, Slice{16, 8}, torch::kFloat32);
  (void)program.EmitCopy(s2, d2);
  (void)program.EmitCopy(s3, d3);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 16], Float)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 16], Float)
  [6] %6 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [16, 8], Float)
  [7] %7 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [16, 8], Float)
  [8] %8 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, Half)
  [9] %9 = pack((%0, %2), %8)
  [10] %10 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, Half)
  [11] %11 = copy(%9, %10)
  [12] (%12, %13) = unpack(%11, (%1, %3))
  [13] %14 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 24, Float)
  [14] %15 = pack((%4, %6), %14)
  [15] %16 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 24, Float)
  [16] %17 = copy(%15, %16)
  [17] (%18, %19) = unpack(%17, (%5, %7))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(PackUnpackCopiesTest, DifferentDtype_SingletonNotGroupedWithOthers) {
  Program program;
  // Two f16 copies (grouped)
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, torch::kFloat16);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, torch::kFloat16);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, torch::kFloat16);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, torch::kFloat16);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  // One f32 copy (singleton, different dtype)
  auto s2 = program.EmitView(dev0, shard, Slice{0, 16}, torch::kFloat32);
  auto d2 = program.EmitView(dev1, shard, Slice{0, 16}, torch::kFloat32);
  (void)program.EmitCopy(s2, d2);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 16], Float)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 16], Float)
  [6] %6 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, Half)
  [7] %7 = pack((%0, %2), %6)
  [8] %8 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, Half)
  [9] %9 = copy(%7, %8)
  [10] (%10, %11) = unpack(%9, (%1, %3))
  [11] %12 = copy(%4, %5)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Mixed same-device and cross-device copies
//==============================================================================

TEST_F(PackUnpackCopiesTest, MixedSameAndCrossDevice_OnlyCrossDeviceGrouped) {
  Program program;
  // Same-device copies (should remain as plain copies)
  auto s_same0 = program.EmitView(dev0, shard, Slice{0, 16}, dt);
  auto d_same0 = program.EmitView(dev0, shard, Slice{16, 16}, dt);
  auto s_same1 = program.EmitView(dev0, shard, Slice{32, 16}, dt);
  auto d_same1 = program.EmitView(dev0, shard, Slice{48, 16}, dt);
  (void)program.EmitCopy(s_same0, d_same0);
  (void)program.EmitCopy(s_same1, d_same1);

  // Cross-device copies (should be grouped)
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, dt);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 16], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [16, 16], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [32, 16], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [48, 16], Half)
  [4] %4 = copy(%0, %1)
  [5] %5 = copy(%2, %3)
  [6] %6 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [7] %7 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [8] %8 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [9] %9 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [10] %10 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, Half)
  [11] %11 = pack((%6, %8), %10)
  [12] %12 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, Half)
  [13] %13 = copy(%11, %12)
  [14] (%14, %15) = unpack(%13, (%7, %9))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Scaling: many copies in one group
//==============================================================================

TEST_F(PackUnpackCopiesTest, ManyCopies_AllPackedIntoOneGroup) {
  // 10 cross-device copies should all be packed into a single group.
  constexpr std::size_t kNumCopies = 10;
  constexpr std::size_t kElemSize = 32;

  Program program;
  for (std::size_t i = 0; i < kNumCopies; ++i) {
    auto src =
        program.EmitView(dev0, shard, Slice{i * kElemSize, kElemSize}, dt);
    auto dst =
        program.EmitView(dev1, shard, Slice{i * kElemSize, kElemSize}, dt);
    (void)program.EmitCopy(src, dst);
  }

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 32], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 32], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [32, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [32, 32], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [6] %6 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [96, 32], Half)
  [7] %7 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [96, 32], Half)
  [8] %8 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [128, 32], Half)
  [9] %9 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [128, 32], Half)
  [10] %10 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [160, 32], Half)
  [11] %11 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [160, 32], Half)
  [12] %12 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [192, 32], Half)
  [13] %13 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [192, 32], Half)
  [14] %14 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [224, 32], Half)
  [15] %15 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [224, 32], Half)
  [16] %16 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [256, 32], Half)
  [17] %17 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [256, 32], Half)
  [18] %18 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [288, 32], Half)
  [19] %19 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [288, 32], Half)
  [20] %20 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 320, Half)
  [21] %21 = pack((%0, %2, %4, %6, %8, %10, %12, %14, %16, %18), %20)
  [22] %22 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 320, Half)
  [23] %23 = copy(%21, %22)
  [24] (%24, %25, %26, %27, %28, %29, %30, %31, %32, %33) = unpack(%23, (%1, %3, %5, %7, %9, %11, %13, %15, %17, %19))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Complex: multiple device pairs + dtypes + same-device + singletons
//==============================================================================

TEST_F(PackUnpackCopiesTest, Complex_AllGroupingRulesApplied) {
  Program program;

  // Group A: 3 copies dev0 → dev1, f16 (should pack)
  for (std::size_t i = 0; i < 3; ++i) {
    auto s = program.EmitView(dev0, shard, Slice{i * 64, 64}, torch::kFloat16);
    auto d = program.EmitView(dev1, shard, Slice{i * 64, 64}, torch::kFloat16);
    (void)program.EmitCopy(s, d);
  }

  // Group B: 2 copies dev0 → dev2, f16 (should pack, different device pair)
  for (std::size_t i = 0; i < 2; ++i) {
    auto s = program.EmitView(dev0, shard, Slice{i * 32, 32}, torch::kFloat16);
    auto d = program.EmitView(dev2, shard, Slice{i * 32, 32}, torch::kFloat16);
    (void)program.EmitCopy(s, d);
  }

  // Group C: 2 copies dev0 → dev1, f32 (should pack, different dtype)
  for (std::size_t i = 0; i < 2; ++i) {
    auto s = program.EmitView(dev0, shard, Slice{i * 16, 16}, torch::kFloat32);
    auto d = program.EmitView(dev1, shard, Slice{i * 16, 16}, torch::kFloat32);
    (void)program.EmitCopy(s, d);
  }

  // Singleton: 1 copy dev1 → dev2, f16 (not packed, only 1)
  auto s_single = program.EmitView(dev1, shard, Slice{0, 48}, torch::kFloat16);
  auto d_single = program.EmitView(dev2, shard, Slice{0, 48}, torch::kFloat16);
  (void)program.EmitCopy(s_single, d_single);

  // Same-device: 1 copy on dev0 (not packed)
  auto s_same = program.EmitView(dev0, shard, Slice{0, 16}, dt);
  auto d_same = program.EmitView(dev0, shard, Slice{16, 16}, dt);
  (void)program.EmitCopy(s_same, d_same);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 64], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 64], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [128, 64], Half)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [128, 64], Half)
  [6] %6 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 32], Half)
  [7] %7 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 32], Half)
  [8] %8 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [32, 32], Half)
  [9] %9 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [32, 32], Half)
  [10] %10 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 16], Float)
  [11] %11 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 16], Float)
  [12] %12 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [16, 16], Float)
  [13] %13 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [16, 16], Float)
  [14] %14 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 48], Half)
  [15] %15 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 48], Half)
  [16] %16 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 16], Half)
  [17] %17 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [16, 16], Half)
  [18] %18 = copy(%16, %17)
  [19] %19 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 192, Half)
  [20] %20 = pack((%0, %2, %4), %19)
  [21] %21 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 192, Half)
  [22] %22 = copy(%20, %21)
  [23] (%23, %24, %25) = unpack(%22, (%1, %3, %5))
  [24] %26 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 32, Float)
  [25] %27 = pack((%10, %12), %26)
  [26] %28 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 32, Float)
  [27] %29 = copy(%27, %28)
  [28] (%30, %31) = unpack(%29, (%11, %13))
  [29] %32 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 64, Half)
  [30] %33 = pack((%6, %8), %32)
  [31] %34 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [32] %35 = copy(%33, %34)
  [33] (%36, %37) = unpack(%35, (%7, %9))
  [34] %38 = copy(%14, %15)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Idempotency: running the pass twice should not change the result
//==============================================================================

TEST_F(PackUnpackCopiesTest, Idempotent_SecondRunNoChange) {
  Program program;
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, dt);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  PackUnpackCopies pass;
  auto first = pass.Run(std::move(program), DefaultCtx());

  auto first_dump = first.Dump();

  auto second = pass.Run(std::move(first), DefaultCtx());

  // After the first pass, there's only 1 cross-device copy (the consolidated
  // one). The second pass should leave it as a singleton — no further packing.
  EXPECT_EQ(second.Dump(), first_dump);
  EXPECT_NO_THROW(Linearity::Check(second));
}

//==============================================================================
// Interleaved copies between different device pairs
//==============================================================================

TEST_F(PackUnpackCopiesTest, InterleavedCopies_GroupedByDevicePair) {
  // Copies to different device pairs interleaved in the program.
  // Should still be grouped correctly by (src_device, dst_device).
  Program program;

  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{0, 48}, dt);
  auto d1 = program.EmitView(dev2, shard, Slice{0, 48}, dt);
  auto s2 = program.EmitView(dev0, shard, Slice{64, 32}, dt);
  auto d2 = program.EmitView(dev1, shard, Slice{64, 32}, dt);
  auto s3 = program.EmitView(dev0, shard, Slice{48, 16}, dt);
  auto d3 = program.EmitView(dev2, shard, Slice{48, 16}, dt);

  // Interleave: dev0→dev1, dev0→dev2, dev0→dev1, dev0→dev2
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);
  (void)program.EmitCopy(s2, d2);
  (void)program.EmitCopy(s3, d3);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 48], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 48], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [6] %6 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [48, 16], Half)
  [7] %7 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [48, 16], Half)
  [8] %8 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, Half)
  [9] %9 = pack((%0, %4), %8)
  [10] %10 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, Half)
  [11] %11 = copy(%9, %10)
  [12] (%12, %13) = unpack(%11, (%1, %5))
  [13] %14 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 64, Half)
  [14] %15 = pack((%2, %6), %14)
  [15] %16 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 64, Half)
  [16] %17 = copy(%15, %16)
  [17] (%18, %19) = unpack(%17, (%3, %7))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Varying source sizes within a group
//==============================================================================

TEST_F(PackUnpackCopiesTest, VaryingSourceSizes_TotalSizeCorrect) {
  // Sources with very different sizes should still produce correct total.
  Program program;
  auto s0 = program.EmitView(dev0, shard, Slice{0, 1}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 1}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{1, 1000}, dt);
  auto d1 = program.EmitView(dev1, shard, Slice{1, 1000}, dt);
  auto s2 = program.EmitView(dev0, shard, Slice{1001, 7}, dt);
  auto d2 = program.EmitView(dev1, shard, Slice{1001, 7}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);
  (void)program.EmitCopy(s2, d2);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [1, 1000], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [1, 1000], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [1001, 7], Half)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [1001, 7], Half)
  [6] %6 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 1008, Half)
  [7] %7 = pack((%0, %2, %4), %6)
  [8] %8 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 1008, Half)
  [9] %9 = copy(%7, %8)
  [10] (%10, %11, %12) = unpack(%9, (%1, %3, %5))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Three devices with all pairwise cross-device copies
//==============================================================================

TEST_F(PackUnpackCopiesTest, ThreeDevices_AllPairsCopied) {
  // 2 copies on each of 3 directed pairs: (0→1), (1→2), (0→2)
  Program program;

  // dev0 → dev1 (2 copies)
  auto s01a = program.EmitView(dev0, shard, Slice{0, 32}, dt);
  auto d01a = program.EmitView(dev1, shard, Slice{0, 32}, dt);
  auto s01b = program.EmitView(dev0, shard, Slice{32, 32}, dt);
  auto d01b = program.EmitView(dev1, shard, Slice{32, 32}, dt);
  (void)program.EmitCopy(s01a, d01a);
  (void)program.EmitCopy(s01b, d01b);

  // dev1 → dev2 (2 copies)
  auto s12a = program.EmitView(dev1, shard, Slice{0, 16}, dt);
  auto d12a = program.EmitView(dev2, shard, Slice{0, 16}, dt);
  auto s12b = program.EmitView(dev1, shard, Slice{16, 16}, dt);
  auto d12b = program.EmitView(dev2, shard, Slice{16, 16}, dt);
  (void)program.EmitCopy(s12a, d12a);
  (void)program.EmitCopy(s12b, d12b);

  // dev0 → dev2 (2 copies)
  auto s02a = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d02a = program.EmitView(dev2, shard, Slice{0, 64}, dt);
  auto s02b = program.EmitView(dev0, shard, Slice{64, 64}, dt);
  auto d02b = program.EmitView(dev2, shard, Slice{64, 64}, dt);
  (void)program.EmitCopy(s02a, d02a);
  (void)program.EmitCopy(s02b, d02b);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 32], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 32], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [32, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [32, 32], Half)
  [4] %4 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 16], Half)
  [5] %5 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 16], Half)
  [6] %6 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [16, 16], Half)
  [7] %7 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [16, 16], Half)
  [8] %8 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [9] %9 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [10] %10 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 64], Half)
  [11] %11 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 64], Half)
  [12] %12 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 64, Half)
  [13] %13 = pack((%0, %2), %12)
  [14] %14 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 64, Half)
  [15] %15 = copy(%13, %14)
  [16] (%16, %17) = unpack(%15, (%1, %3))
  [17] %18 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 128, Half)
  [18] %19 = pack((%8, %10), %18)
  [19] %20 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 128, Half)
  [20] %21 = copy(%19, %20)
  [21] (%22, %23) = unpack(%21, (%9, %11))
  [22] %24 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 32, Half)
  [23] %25 = pack((%4, %6), %24)
  [24] %26 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 32, Half)
  [25] %27 = copy(%25, %26)
  [26] (%28, %29) = unpack(%27, (%5, %7))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Output program must preserve all original ViewOps
//==============================================================================

TEST_F(PackUnpackCopiesTest, ViewOps_Preserved) {
  Program program;
  auto s0 = program.EmitView(dev0, shard, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev1, shard, Slice{0, 64}, dt);
  auto s1 = program.EmitView(dev0, shard, Slice{64, 32}, dt);
  auto d1 = program.EmitView(dev1, shard, Slice{64, 32}, dt);
  (void)program.EmitCopy(s0, d0);
  (void)program.EmitCopy(s1, d1);

  PackUnpackCopies pass;
  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [3] %3 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [64, 32], Half)
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), 96, Half)
  [5] %5 = pack((%0, %2), %4)
  [6] %6 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), 96, Half)
  [7] %7 = copy(%5, %6)
  [8] (%8, %9) = unpack(%7, (%1, %3))
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
}  // namespace
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
