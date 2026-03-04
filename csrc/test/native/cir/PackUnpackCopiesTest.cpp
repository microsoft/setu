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
#include "planner/hints/HintStore.h"
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
#include "planner/passes/PackUnpackCopies.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::Participant;
using setu::planner::hints::HintStore;
using setu::planner::ir::cir::AllocTmpOp;
using setu::planner::ir::cir::CopyOp;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::OpType;
using setu::planner::ir::cir::PackOp;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::UnpackOp;
using setu::planner::ir::cir::Value;
using setu::planner::ir::cir::ValueInfo;
using setu::planner::passes::PackUnpackCopies;
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

  /// Count operations of a given type in a program.
  [[nodiscard]] std::size_t CountOps(const Program& program,
                                     OpType type) const {
    std::size_t count = 0;
    for (const auto& op : program.Operations()) {
      if (op.Type() == type) {
        ++count;
      }
    }
    return count;
  }

  /// Find the first operation of a given type.
  [[nodiscard]] const auto& FindFirstOp(const Program& program,
                                        OpType type) const {
    for (const auto& op : program.Operations()) {
      if (op.Type() == type) {
        return op;
      }
    }
    EXPECT_TRUE(false) << "No op of requested type found";
    return program.Operations().front();  // unreachable
  }

  /// Collect all operations of a given type.
  [[nodiscard]] std::vector<
      std::reference_wrapper<const setu::planner::ir::cir::Operation>>
  FindAllOps(const Program& program, OpType type) const {
    std::vector<std::reference_wrapper<const setu::planner::ir::cir::Operation>>
        result;
    for (const auto& op : program.Operations()) {
      if (op.Type() == type) {
        result.push_back(std::cref(op));
      }
    }
    return result;
  }
};

//==============================================================================
// Empty / no-op cases
//==============================================================================

TEST_F(PackUnpackCopiesTest, EmptyProgram_ProducesEmptyProgram) {
  Program program;
  PackUnpackCopies pass;
  auto result = pass.Run(program, hints);

  EXPECT_EQ(result.NumOperations(), 0u);
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(PackUnpackCopiesTest, ViewsOnly_NoCopies_PassedThrough) {
  // Program with only views and no copies should be unchanged.
  Program program;
  (void)program.EmitView(dev0, shard, Slice{0, 64}, dt);
  (void)program.EmitView(dev1, shard, Slice{0, 32}, dt);

  PackUnpackCopies pass;
  auto result = pass.Run(program, hints);

  EXPECT_EQ(CountOps(result, OpType::kView), 2u);
  EXPECT_EQ(CountOps(result, OpType::kCopy), 0u);
  EXPECT_EQ(CountOps(result, OpType::kPack), 0u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 0u);
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
  auto result = pass.Run(program, hints);

  EXPECT_EQ(CountOps(result, OpType::kCopy), 2u);
  EXPECT_EQ(CountOps(result, OpType::kPack), 0u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 0u);
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 0u);
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
  auto result = pass.Run(program, hints);

  EXPECT_EQ(CountOps(result, OpType::kCopy), 1u);
  EXPECT_EQ(CountOps(result, OpType::kPack), 0u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 0u);
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 0u);
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
  auto result = pass.Run(program, hints);

  // Two individual copies replaced by: alloc_tmp, pack, alloc_tmp, copy, unpack
  EXPECT_EQ(CountOps(result, OpType::kPack), 1u);
  EXPECT_EQ(CountOps(result, OpType::kCopy), 1u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 1u);
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 2u);
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
  auto result = pass.Run(program, hints);

  const std::size_t expected_total = 64 + 32;
  auto alloc_ops = FindAllOps(result, OpType::kAllocTmp);
  ASSERT_EQ(alloc_ops.size(), 2u);
  for (const auto& op_ref : alloc_ops) {
    const auto& alloc = std::get<AllocTmpOp>(op_ref.get().op);
    EXPECT_EQ(alloc.size_elements, expected_total)
        << "Temp buffer size should equal total source size";
  }
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
  auto result = pass.Run(program, hints);

  auto alloc_ops = FindAllOps(result, OpType::kAllocTmp);
  ASSERT_EQ(alloc_ops.size(), 2u);

  std::set<Device> alloc_devices;
  for (const auto& op_ref : alloc_ops) {
    const auto& alloc = std::get<AllocTmpOp>(op_ref.get().op);
    auto info = result.GetValueInfo(alloc.out);
    alloc_devices.insert(info.device);
  }
  EXPECT_TRUE(alloc_devices.contains(dev0))
      << "Should allocate a temp on the source device";
  EXPECT_TRUE(alloc_devices.contains(dev1))
      << "Should allocate a temp on the destination device";
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
  auto result = pass.Run(program, hints);

  auto alloc_ops = FindAllOps(result, OpType::kAllocTmp);
  ASSERT_EQ(alloc_ops.size(), 2u);
  for (const auto& op_ref : alloc_ops) {
    const auto& alloc = std::get<AllocTmpOp>(op_ref.get().op);
    auto info = result.GetValueInfo(alloc.out);
    EXPECT_EQ(info.dtype, dtype)
        << "Temp buffer dtype should match source dtype";
  }
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
  auto result = pass.Run(program, hints);

  EXPECT_EQ(CountOps(result, OpType::kPack), 1u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 1u);

  const auto& pack_op = std::get<PackOp>(FindFirstOp(result, OpType::kPack).op);
  EXPECT_EQ(pack_op.srcs.size(), 3u)
      << "Pack should have one source per grouped copy";

  const auto& unpack_op =
      std::get<UnpackOp>(FindFirstOp(result, OpType::kUnpack).op);
  EXPECT_EQ(unpack_op.dst_ins.size(), 3u)
      << "Unpack should have one destination per grouped copy";
  EXPECT_EQ(unpack_op.dst_outs.size(), 3u)
      << "Unpack should produce one output per grouped copy";

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
  auto result = pass.Run(program, hints);

  // Each device pair produces its own pack/copy/unpack
  EXPECT_EQ(CountOps(result, OpType::kPack), 2u);
  EXPECT_EQ(CountOps(result, OpType::kCopy), 2u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 2u);
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 4u);
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
  auto result = pass.Run(program, hints);

  // dev0→dev1 and dev1→dev0 are distinct device pairs → 2 separate groups
  EXPECT_EQ(CountOps(result, OpType::kPack), 2u);
  EXPECT_EQ(CountOps(result, OpType::kCopy), 2u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 2u);
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 4u);
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
  auto result = pass.Run(program, hints);

  // Two dtype groups → 2 packs, 2 copies, 2 unpacks
  EXPECT_EQ(CountOps(result, OpType::kPack), 2u);
  EXPECT_EQ(CountOps(result, OpType::kCopy), 2u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 2u);
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
  auto result = pass.Run(program, hints);

  // f16 group: 1 pack + 1 copy + 1 unpack
  // f32 singleton: 1 plain copy
  EXPECT_EQ(CountOps(result, OpType::kPack), 1u);
  EXPECT_EQ(CountOps(result, OpType::kCopy), 2u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 1u);
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
  auto result = pass.Run(program, hints);

  // 2 same-device copies + 1 consolidated cross-device copy = 3 CopyOps
  EXPECT_EQ(CountOps(result, OpType::kCopy), 3u);
  EXPECT_EQ(CountOps(result, OpType::kPack), 1u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 1u);
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
  auto result = pass.Run(program, hints);

  EXPECT_EQ(CountOps(result, OpType::kPack), 1u);
  EXPECT_EQ(CountOps(result, OpType::kCopy), 1u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 1u);

  // Verify pack has correct number of sources
  const auto& pack_op = std::get<PackOp>(FindFirstOp(result, OpType::kPack).op);
  EXPECT_EQ(pack_op.srcs.size(), kNumCopies);

  // Verify temp buffer size = sum of all source sizes
  auto alloc_ops = FindAllOps(result, OpType::kAllocTmp);
  for (const auto& op_ref : alloc_ops) {
    const auto& alloc = std::get<AllocTmpOp>(op_ref.get().op);
    EXPECT_EQ(alloc.size_elements, kNumCopies * kElemSize);
  }

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
  auto result = pass.Run(program, hints);

  // 3 groups packed (A, B, C)
  EXPECT_EQ(CountOps(result, OpType::kPack), 3u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 3u);

  // 3 consolidated copies + 1 singleton + 1 same-device = 5 copies
  EXPECT_EQ(CountOps(result, OpType::kCopy), 5u);

  // 3 groups × 2 alloc_tmp each = 6
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 6u);

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
  auto first = pass.Run(program, hints);
  auto second = pass.Run(first, hints);

  // After the first pass, there's only 1 cross-device copy (the consolidated
  // one). The second pass should leave it as a singleton — no further packing.
  EXPECT_EQ(CountOps(second, OpType::kPack), CountOps(first, OpType::kPack));
  EXPECT_EQ(CountOps(second, OpType::kCopy), CountOps(first, OpType::kCopy));
  EXPECT_EQ(CountOps(second, OpType::kUnpack),
            CountOps(first, OpType::kUnpack));
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
  auto result = pass.Run(program, hints);

  // Should produce 2 groups, each with 2 copies consolidated
  EXPECT_EQ(CountOps(result, OpType::kPack), 2u);
  EXPECT_EQ(CountOps(result, OpType::kCopy), 2u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 2u);
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
  auto result = pass.Run(program, hints);

  const std::size_t expected_total = 1 + 1000 + 7;
  auto alloc_ops = FindAllOps(result, OpType::kAllocTmp);
  ASSERT_EQ(alloc_ops.size(), 2u);
  for (const auto& op_ref : alloc_ops) {
    const auto& alloc = std::get<AllocTmpOp>(op_ref.get().op);
    EXPECT_EQ(alloc.size_elements, expected_total);
  }

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
  auto result = pass.Run(program, hints);

  // 3 device pairs → 3 groups, each consolidated
  EXPECT_EQ(CountOps(result, OpType::kPack), 3u);
  EXPECT_EQ(CountOps(result, OpType::kCopy), 3u);
  EXPECT_EQ(CountOps(result, OpType::kUnpack), 3u);
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 6u);
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
  auto result = pass.Run(program, hints);

  EXPECT_EQ(CountOps(result, OpType::kView), 4u)
      << "All original ViewOps should be preserved";
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
}  // namespace
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
