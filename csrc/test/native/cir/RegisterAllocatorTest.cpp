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
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::LivenessInfo;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::RegisterAllocation;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::Value;
//==============================================================================
namespace {
//==============================================================================

/// Helper to create a Device for testing (node0, cuda:0)
Device MakeDevice(std::int16_t gpu_index = 0) {
  auto node_id = boost::uuids::nil_uuid();
  return Device(node_id, setu::commons::datatypes::Device(torch::Device(
                             torch::kCUDA, static_cast<int8_t>(gpu_index))));
}

/// Helper to create a second-node device for multi-device tests
Device MakeDevice(boost::uuids::uuid node_id, std::int16_t gpu_index = 0) {
  return Device(node_id, setu::commons::datatypes::Device(torch::Device(
                             torch::kCUDA, static_cast<int8_t>(gpu_index))));
}

/// Helper to create a ShardRef for view ops
setu::planner::ir::ref::ShardRef MakeShardRef() {
  return setu::planner::ir::ref::ShardRef(boost::uuids::nil_uuid());
}

//==============================================================================
// Single allocation
//==============================================================================

TEST(CIRRegisterAllocatorTest, SingleAllocTmp_AssignsRegisterZero) {
  // Program: %0 = alloc_tmp(dev, 128, f16)
  Program program;
  auto dev = MakeDevice();
  auto v0 = program.EmitAllocTmp(dev, 128, torch::kFloat16);

  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, std::uint32_t> pool_sizes = {{dev, 4}};
  auto alloc = RegisterAllocation::Build(program, liveness, pool_sizes);

  ASSERT_TRUE(alloc.allocation[v0.id].has_value())
      << "AllocTmp value should be assigned a physical register";
  EXPECT_EQ(alloc.allocation[v0.id]->register_index, 0u)
      << "First allocation should get register 0";
  EXPECT_EQ(alloc.allocation[v0.id]->device, dev);
}

//==============================================================================
// Non-overlapping live ranges reuse registers
//==============================================================================

TEST(CIRRegisterAllocatorTest, NonOverlapping_ReusesSameRegister) {
  // Program:
  //   [0] %0 = alloc_tmp(dev, 64, f16)
  //   [1] %1 = alloc_tmp(dev, 64, f16)
  //   [2] %2 = copy(%0, %1)        -- uses %0 and %1, %0's last use
  //   [3] %3 = alloc_tmp(dev, 64, f16)  -- %0 is dead, its register is free
  //
  // %0 live range: [0, 2], %3 live range: [3, 3]
  // Since %0 is dead after op 2, register 0 can be reused for %3.
  Program program;
  auto dev = MakeDevice();

  auto v0 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto v1 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitCopy(v0, v1);  // uses v0 and v1
  auto v3 = program.EmitAllocTmp(dev, 64, torch::kFloat16);

  auto liveness = LivenessInfo::Build(program);

  // Only 2 slots in the pool -- enough if reuse works
  std::unordered_map<Device, std::uint32_t> pool_sizes = {{dev, 2}};
  auto alloc = RegisterAllocation::Build(program, liveness, pool_sizes);

  ASSERT_TRUE(alloc.allocation[v0.id].has_value());
  ASSERT_TRUE(alloc.allocation[v1.id].has_value());
  ASSERT_TRUE(alloc.allocation[v3.id].has_value());

  // v0 and v3 should share a register since their live ranges don't overlap
  EXPECT_EQ(alloc.allocation[v0.id]->register_index,
            alloc.allocation[v3.id]->register_index)
      << "Non-overlapping live ranges should reuse the same register";

  // v0 and v1 must have different registers (both live at op 2)
  EXPECT_NE(alloc.allocation[v0.id]->register_index,
            alloc.allocation[v1.id]->register_index)
      << "Overlapping live ranges must use different registers";
}

//==============================================================================
// Overlapping live ranges get distinct registers
//==============================================================================

TEST(CIRRegisterAllocatorTest, Overlapping_AssignsDistinctRegisters) {
  // Program:
  //   [0] %0 = alloc_tmp(dev, 64, f16)
  //   [1] %1 = alloc_tmp(dev, 64, f16)
  //   [2] %2 = alloc_tmp(dev, 64, f16)
  //   [3] %3 = copy(%0, %2)        -- uses %0 and %2
  //   [4] %4 = copy(%1, %3)        -- uses %1 and %3
  //
  // %0 is live [0, 3], %1 is live [1, 4], %2 is live [2, 3].
  // All three overlap around ops 2-3, so they need 3 distinct registers.
  Program program;
  auto dev = MakeDevice();

  auto v0 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto v1 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto v2 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto v3 = program.EmitCopy(v0, v2);
  (void)program.EmitCopy(v1, v3);

  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, std::uint32_t> pool_sizes = {{dev, 4}};
  auto alloc = RegisterAllocation::Build(program, liveness, pool_sizes);

  ASSERT_TRUE(alloc.allocation[v0.id].has_value());
  ASSERT_TRUE(alloc.allocation[v1.id].has_value());
  ASSERT_TRUE(alloc.allocation[v2.id].has_value());

  // All three must have distinct register indices
  std::set<std::uint32_t> indices = {
      alloc.allocation[v0.id]->register_index,
      alloc.allocation[v1.id]->register_index,
      alloc.allocation[v2.id]->register_index,
  };
  EXPECT_EQ(indices.size(), 3u)
      << "Three simultaneously live AllocTmp values need 3 distinct registers";
}

//==============================================================================
// Multi-device: independent register pools
//==============================================================================

TEST(CIRRegisterAllocatorTest, MultiDevice_IndependentPools) {
  // Two devices, each with one AllocTmp. Each device has its own pool.
  Program program;
  auto dev_a = MakeDevice(0);
  auto dev_b = MakeDevice(1);

  auto va = program.EmitAllocTmp(dev_a, 64, torch::kFloat16);
  auto vb = program.EmitAllocTmp(dev_b, 64, torch::kFloat16);

  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, std::uint32_t> pool_sizes = {{dev_a, 2},
                                                          {dev_b, 2}};
  auto alloc = RegisterAllocation::Build(program, liveness, pool_sizes);

  ASSERT_TRUE(alloc.allocation[va.id].has_value());
  ASSERT_TRUE(alloc.allocation[vb.id].has_value());

  // Both get register 0 from their respective pools
  EXPECT_EQ(alloc.allocation[va.id]->register_index, 0u);
  EXPECT_EQ(alloc.allocation[vb.id]->register_index, 0u);

  // Devices match
  EXPECT_EQ(alloc.allocation[va.id]->device, dev_a);
  EXPECT_EQ(alloc.allocation[vb.id]->device, dev_b);
}

//==============================================================================
// View-only values are not allocated
//==============================================================================

TEST(CIRRegisterAllocatorTest, ViewValues_NotAllocated) {
  // Program:
  //   [0] %0 = view(dev, shard, [0,128], f16)
  //   [1] %1 = alloc_tmp(dev, 128, f16)
  //   [2] %2 = copy(%0, %1)
  Program program;
  auto dev = MakeDevice();
  auto shard = MakeShardRef();

  auto v_view = program.EmitView(dev, shard, Slice{0, 128}, torch::kFloat16);
  auto v_tmp = program.EmitAllocTmp(dev, 128, torch::kFloat16);
  auto v_copy = program.EmitCopy(v_view, v_tmp);

  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, std::uint32_t> pool_sizes = {{dev, 4}};
  auto alloc = RegisterAllocation::Build(program, liveness, pool_sizes);

  // View value should not be allocated a physical register
  EXPECT_FALSE(alloc.allocation[v_view.id].has_value())
      << "View-defined value should not get a physical register";

  // Copy result should not be allocated a physical register
  EXPECT_FALSE(alloc.allocation[v_copy.id].has_value())
      << "Copy-defined value should not get a physical register";

  // AllocTmp value should be allocated
  ASSERT_TRUE(alloc.allocation[v_tmp.id].has_value())
      << "AllocTmp-defined value should get a physical register";
}

//==============================================================================
// Register reuse across a chain of temporaries
//==============================================================================

TEST(CIRRegisterAllocatorTest, Chain_ReusesRegistersSequentially) {
  // A chain where each temp is used once then dead:
  //   [0] %0 = alloc_tmp(dev, 64, f16)
  //   [1] %1 = view(dev, shard, [0,64], f16)
  //   [2] %2 = copy(%1, %0)          -- %0 last use here
  //   [3] %3 = alloc_tmp(dev, 64, f16)  -- %0's register is available
  //   [4] %4 = view(dev, shard, [64,64], f16)
  //   [5] %5 = copy(%4, %3)          -- %3 last use here
  //   [6] %6 = alloc_tmp(dev, 64, f16)  -- %3's register is available
  //
  // With pool_size=1, all three AllocTmps should reuse register 0.
  Program program;
  auto dev = MakeDevice();
  auto shard = MakeShardRef();

  auto t0 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto v1 = program.EmitView(dev, shard, Slice{0, 64}, torch::kFloat16);
  (void)program.EmitCopy(v1, t0);

  auto t3 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto v4 = program.EmitView(dev, shard, Slice{64, 64}, torch::kFloat16);
  (void)program.EmitCopy(v4, t3);

  auto t6 = program.EmitAllocTmp(dev, 64, torch::kFloat16);

  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, std::uint32_t> pool_sizes = {{dev, 1}};
  auto alloc = RegisterAllocation::Build(program, liveness, pool_sizes);

  ASSERT_TRUE(alloc.allocation[t0.id].has_value());
  ASSERT_TRUE(alloc.allocation[t3.id].has_value());
  ASSERT_TRUE(alloc.allocation[t6.id].has_value());

  // All should share register 0 since they never overlap
  EXPECT_EQ(alloc.allocation[t0.id]->register_index, 0u);
  EXPECT_EQ(alloc.allocation[t3.id]->register_index, 0u);
  EXPECT_EQ(alloc.allocation[t6.id]->register_index, 0u);
}

//==============================================================================
// Pool exhaustion asserts
//==============================================================================

TEST(CIRRegisterAllocatorTest, PoolExhausted_Asserts) {
  // Two simultaneously live AllocTmps but pool of size 1 -- should assert.
  Program program;
  auto dev = MakeDevice();

  auto v0 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto v1 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  // Both v0 and v1 are used in the copy -> both live at op 2
  (void)program.EmitCopy(v0, v1);

  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, std::uint32_t> pool_sizes = {{dev, 1}};

  EXPECT_THROW(RegisterAllocation::Build(program, liveness, pool_sizes),
               std::runtime_error)
      << "Should throw when pool cannot satisfy simultaneous live registers";
}

//==============================================================================
// Empty program
//==============================================================================

TEST(CIRRegisterAllocatorTest, EmptyProgram_ProducesEmptyAllocation) {
  Program program;
  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, std::uint32_t> pool_sizes;
  auto alloc = RegisterAllocation::Build(program, liveness, pool_sizes);

  EXPECT_TRUE(alloc.allocation.empty());
}

//==============================================================================
}  // namespace
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
