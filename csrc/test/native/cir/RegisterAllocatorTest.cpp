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
#include "planner/RegisterSet.h"
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::RegisterSet;
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
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev, RegisterSet::Uniform(4, 1024)}};
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

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

  // Only 2 registers in the set -- enough if reuse works
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev, RegisterSet::Uniform(2, 1024)}};
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

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
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev, RegisterSet::Uniform(4, 1024)}};
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

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
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev_a, RegisterSet::Uniform(2, 1024)},
      {dev_b, RegisterSet::Uniform(2, 1024)}};
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

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
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev, RegisterSet::Uniform(4, 1024)}};
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

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
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev, RegisterSet::Uniform(1, 1024)}};
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

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
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev, RegisterSet::Uniform(1, 1024)}};

  EXPECT_THROW(RegisterAllocation::Build(program, liveness, register_sets),
               std::runtime_error)
      << "Should throw when pool cannot satisfy simultaneous live registers";
}

//==============================================================================
// Empty program
//==============================================================================

TEST(CIRRegisterAllocatorTest, EmptyProgram_ProducesEmptyAllocation) {
  Program program;
  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, RegisterSet> register_sets;
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

  EXPECT_TRUE(alloc.allocation.empty());
}

//==============================================================================
// Alias chain: liveness extended through CopyOp alias
//==============================================================================

TEST(CIRRegisterAllocatorTest, AliasChain_ExtendedLiveness_DistinctRegisters) {
  // A→B→C relay where B has two AllocTmpOps. The first tmp's alias chain
  // (through CopyOp's dst_out and SliceOp) overlaps with the second tmp's
  // live range. Without alias-aware liveness, r0 and r1 would share a
  // register, corrupting memory.
  //
  //   [0] %v0 = view(A)
  //   [1] %r0 = alloc_tmp(B)
  //   [2] %r0' = copy(%v0, %r0)     -- alias: r0' shares r0's physical memory
  //   [3] %s0 = slice(%r0')          -- alias: s0 shares r0's physical memory
  //   [4] %r1 = alloc_tmp(B)
  //   [5] %r1' = copy(%s0, %r1)     -- reads from r0's memory → r0 still live
  //
  // r0 and r1 must get distinct registers.
  auto node_a =
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000001");
  auto node_b =
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000002");
  auto dev_a = MakeDevice(node_a, 0);
  auto dev_b = MakeDevice(node_b, 0);

  Program program;
  auto v0 =
      program.EmitView(dev_a, MakeShardRef(), Slice{0, 128}, torch::kFloat16);
  auto r0 = program.EmitAllocTmp(dev_b, 128, torch::kFloat16);
  auto r0_out = program.EmitCopy(v0, r0);
  auto s0 = program.EmitSlice(r0_out, Slice{0, 64});
  auto r1 = program.EmitAllocTmp(dev_b, 64, torch::kFloat16);
  (void)program.EmitCopy(s0, r1);

  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev_b, RegisterSet::Uniform(4, 1024)}};
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

  ASSERT_TRUE(alloc.allocation[r0.id].has_value());
  ASSERT_TRUE(alloc.allocation[r1.id].has_value());
  EXPECT_NE(alloc.allocation[r0.id]->register_index,
            alloc.allocation[r1.id]->register_index)
      << "Alias chain should extend r0's liveness, forcing distinct registers";
}

//==============================================================================
// Alias chain: ConsumeOp extends liveness
//==============================================================================

TEST(CIRRegisterAllocatorTest, AliasChain_ConsumeOp_ExtendsLiveness) {
  // ConsumeOp creates an alias: out shares src's physical memory.
  //   [0] %r0 = alloc_tmp(dev)
  //   [1] %c0 = consume(%r0)           -- alias: c0 shares r0's memory
  //   [2] %r1 = alloc_tmp(dev)
  //   [3] %r1' = copy(%c0, %r1)        -- reads r0's memory
  //
  // r0's liveness must extend through c0's last use (op 3).
  auto dev = MakeDevice();

  Program program;
  auto r0 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto c0 = program.EmitConsume(r0);
  auto r1 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitCopy(c0, r1);

  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev, RegisterSet::Uniform(4, 1024)}};
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

  ASSERT_TRUE(alloc.allocation[r0.id].has_value());
  ASSERT_TRUE(alloc.allocation[r1.id].has_value());
  EXPECT_NE(alloc.allocation[r0.id]->register_index,
            alloc.allocation[r1.id]->register_index)
      << "ConsumeOp alias should extend r0's liveness";
}

//==============================================================================
// Alias chain: deep relay with distinct registers
//==============================================================================

TEST(CIRRegisterAllocatorTest, AliasChain_DeepRelay_DistinctRegisters) {
  // 4-hop relay A→B→C→D→E with AllocTmpOps on B, C, D.
  // Each hop: copy into tmp, slice, copy to next tmp.
  // Alias chains extend through slices, ensuring overlapping tmps on the
  // same device get distinct registers.
  auto node_a =
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000001");
  auto node_b =
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000002");
  auto node_c =
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000003");
  auto node_d =
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000004");
  auto node_e =
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000005");
  auto dev_a = MakeDevice(node_a, 0);
  auto dev_b = MakeDevice(node_b, 0);
  auto dev_c = MakeDevice(node_c, 0);
  auto dev_d = MakeDevice(node_d, 0);
  auto dev_e = MakeDevice(node_e, 0);

  Program program;
  auto src =
      program.EmitView(dev_a, MakeShardRef(), Slice{0, 64}, torch::kFloat16);
  auto dst =
      program.EmitView(dev_e, MakeShardRef(), Slice{0, 64}, torch::kFloat16);

  auto tb = program.EmitAllocTmp(dev_b, 64, torch::kFloat16);
  auto tb_out = program.EmitCopy(src, tb);

  auto tc = program.EmitAllocTmp(dev_c, 64, torch::kFloat16);
  auto tc_out = program.EmitCopy(tb_out, tc);

  auto td = program.EmitAllocTmp(dev_d, 64, torch::kFloat16);
  auto td_out = program.EmitCopy(tc_out, td);

  (void)program.EmitCopy(td_out, dst);

  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev_b, RegisterSet::Uniform(4, 1024)},
      {dev_c, RegisterSet::Uniform(4, 1024)},
      {dev_d, RegisterSet::Uniform(4, 1024)}};
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

  ASSERT_TRUE(alloc.allocation[tb.id].has_value());
  ASSERT_TRUE(alloc.allocation[tc.id].has_value());
  ASSERT_TRUE(alloc.allocation[td.id].has_value());

  // Each device only has one tmp, so each should get register 0
  EXPECT_EQ(alloc.allocation[tb.id]->register_index, 0u);
  EXPECT_EQ(alloc.allocation[tc.id]->register_index, 0u);
  EXPECT_EQ(alloc.allocation[td.id]->register_index, 0u);
}

//==============================================================================
// Alias chain: non-overlapping chains still reuse registers
//==============================================================================

TEST(CIRRegisterAllocatorTest, AliasChain_NonOverlapping_StillReuses) {
  // Two sequential relay chains through the same device. The first chain's
  // alias is fully consumed before the second starts.
  //
  //   [0] %src0 = view(A)
  //   [1] %dst0 = view(C)
  //   [2] %r0 = alloc_tmp(B)
  //   [3] %r0' = copy(%src0, %r0)
  //   [4] %_ = copy(%r0', %dst0)       -- r0's alias chain fully consumed
  //   [5] %src1 = view(A)
  //   [6] %dst1 = view(C)
  //   [7] %r1 = alloc_tmp(B)           -- r0's register should be reusable
  //   [8] %r1' = copy(%src1, %r1)
  //   [9] %_ = copy(%r1', %dst1)
  auto node_a =
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000001");
  auto node_b =
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000002");
  auto node_c =
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000003");
  auto dev_a = MakeDevice(node_a, 0);
  auto dev_b = MakeDevice(node_b, 0);
  auto dev_c = MakeDevice(node_c, 0);

  Program program;
  auto src0 =
      program.EmitView(dev_a, MakeShardRef(), Slice{0, 64}, torch::kFloat16);
  auto dst0 =
      program.EmitView(dev_c, MakeShardRef(), Slice{0, 64}, torch::kFloat16);
  auto r0 = program.EmitAllocTmp(dev_b, 64, torch::kFloat16);
  auto r0_out = program.EmitCopy(src0, r0);
  (void)program.EmitCopy(r0_out, dst0);

  auto src1 =
      program.EmitView(dev_a, MakeShardRef(), Slice{64, 64}, torch::kFloat16);
  auto dst1 =
      program.EmitView(dev_c, MakeShardRef(), Slice{64, 64}, torch::kFloat16);
  auto r1 = program.EmitAllocTmp(dev_b, 64, torch::kFloat16);
  auto r1_out = program.EmitCopy(src1, r1);
  (void)program.EmitCopy(r1_out, dst1);

  auto liveness = LivenessInfo::Build(program);
  std::unordered_map<Device, RegisterSet> register_sets = {
      {dev_b, RegisterSet::Uniform(1, 1024)}};
  auto alloc = RegisterAllocation::Build(program, liveness, register_sets);

  ASSERT_TRUE(alloc.allocation[r0.id].has_value());
  ASSERT_TRUE(alloc.allocation[r1.id].has_value());
  EXPECT_EQ(alloc.allocation[r0.id]->register_index,
            alloc.allocation[r1.id]->register_index)
      << "Non-overlapping alias chains should reuse the same register";
}

//==============================================================================
}  // namespace
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
