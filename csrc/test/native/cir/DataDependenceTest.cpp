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
#include "planner/ir/ref/ShardRef.h"
#include "planner/targets/DataDependence.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::LivenessInfo;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::RegisterAllocation;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::Value;
using setu::planner::ir::ref::ShardRef;
using setu::planner::targets::BuildDataDependence;
using setu::planner::targets::DataDependence;
//==============================================================================
namespace {

Device MakeTestDevice(std::int16_t gpu_index = 0) {
  auto node_id = boost::uuids::nil_uuid();
  return Device(node_id, setu::commons::datatypes::Device(torch::Device(
                             torch::kCUDA, static_cast<int8_t>(gpu_index))));
}

ShardRef MakeShardA() {
  return ShardRef(
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000001"));
}
ShardRef MakeShardB() {
  return ShardRef(
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000002"));
}
ShardRef MakeShardC() {
  return ShardRef(
      boost::uuids::string_generator()("00000000-0000-0000-0000-000000000003"));
}

}  // namespace
//==============================================================================

class DataDependenceTest : public ::testing::Test {
 protected:
  Device dev0 = MakeTestDevice(0);
  Device dev1 = MakeTestDevice(1);
  ShardRef shard_a = MakeShardA();
  ShardRef shard_b = MakeShardB();
  ShardRef shard_c = MakeShardC();
  torch::Dtype dt = torch::kFloat32;  // 4 bytes per element
};

//==============================================================================
// Structural: reference builders do not appear as nodes
//==============================================================================

TEST_F(DataDependenceTest, EmptyProgram_EmptyDag) {
  Program program;
  auto dag = BuildDataDependence(program, std::nullopt);
  EXPECT_TRUE(dag.nodes.empty());
  EXPECT_TRUE(dag.preds.empty());
}

TEST_F(DataDependenceTest, OnlyViews_NoNodes) {
  Program program;
  (void)program.EmitView(dev0, shard_a, Slice{0, 128}, dt);
  (void)program.EmitView(dev1, shard_b, Slice{0, 128}, dt);
  auto dag = BuildDataDependence(program, std::nullopt);
  EXPECT_TRUE(dag.nodes.empty());
}

TEST_F(DataDependenceTest, ViewAndSliceAndConsume_NoNodes) {
  Program program;
  auto v = program.EmitView(dev0, shard_a, Slice{0, 128}, dt);
  auto s = program.EmitSlice(v, Slice{0, 64});
  (void)program.EmitConsume(s);
  auto dag = BuildDataDependence(program, std::nullopt);
  EXPECT_TRUE(dag.nodes.empty());
}

//==============================================================================
// Single data-moving op
//==============================================================================

TEST_F(DataDependenceTest, SingleCopy_OneNodeNoPreds) {
  Program program;
  auto src = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto dst = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  (void)program.EmitCopy(src, dst);

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 1u);
  EXPECT_EQ(dag.nodes[0].reads.size(), 1u);
  EXPECT_EQ(dag.nodes[0].writes.size(), 1u);
  EXPECT_EQ(dag.nodes[0].participants.size(), 2u)
      << "cross-device copy has two participants";
  ASSERT_EQ(dag.preds.size(), 1u);
  EXPECT_TRUE(dag.preds[0].empty());
}

//==============================================================================
// RAW / WAW / WAR edges
//==============================================================================

TEST_F(DataDependenceTest, RawEdge_SecondCopyReadsFirstsWrite) {
  // %0 = view dev0 shard_a [0, 64)
  // %1 = view dev1 shard_b [0, 64)
  // copy(%0, %1)                 ← writes (dev1, shard_b, [0, 64))
  // %2 = view dev1 shard_b [0, 64)
  // %3 = view dev0 shard_a [0, 64)
  // copy(%2, %3)                 ← reads (dev1, shard_b, [0, 64))
  Program program;
  auto a = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto b = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  (void)program.EmitCopy(a, b);
  auto b2 = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  auto a2 = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  (void)program.EmitCopy(b2, a2);

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 2u);
  EXPECT_TRUE(dag.preds[0].empty());
  EXPECT_EQ(dag.preds[1], std::set<std::uint32_t>{0});
}

TEST_F(DataDependenceTest, WawEdge_SecondCopyOverwritesFirst) {
  // Two copies that both write (dev1, shard_b, [0, 64)).
  Program program;
  auto a = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto b1 = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  (void)program.EmitCopy(a, b1);
  auto a2 = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto b2 = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  (void)program.EmitCopy(a2, b2);

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 2u);
  EXPECT_TRUE(dag.preds[0].empty());
  EXPECT_EQ(dag.preds[1], std::set<std::uint32_t>{0});
}

TEST_F(DataDependenceTest, WarEdge_SecondWriteAfterFirstRead) {
  // copy(%0, %1): reads (dev0, shard_a, [0, 64))
  // copy(%2, %3): writes (dev0, shard_a, [0, 64))   — WAR
  Program program;
  auto a = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto b = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  (void)program.EmitCopy(a, b);
  auto b2 = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  auto a2 = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  (void)program.EmitCopy(b2, a2);

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 2u);
  EXPECT_TRUE(dag.preds[0].empty());
  // First node read (dev0, shard_a, [0,64)); second node writes it. WAR edge.
  EXPECT_EQ(dag.preds[1], std::set<std::uint32_t>{0});
}

//==============================================================================
// No edges when regions don't overlap
//==============================================================================

TEST_F(DataDependenceTest, DisjointWrites_NoEdge) {
  // Two copies to disjoint offsets of the same (participant, buffer_ref).
  Program program;
  auto a1 = program.EmitView(dev0, shard_a, Slice{0, 16}, dt);
  auto b1 = program.EmitView(dev1, shard_b, Slice{0, 16}, dt);
  (void)program.EmitCopy(a1, b1);
  auto a2 = program.EmitView(dev0, shard_a, Slice{32, 16}, dt);
  auto b2 = program.EmitView(dev1, shard_b, Slice{32, 16}, dt);
  (void)program.EmitCopy(a2, b2);

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 2u);
  EXPECT_TRUE(dag.preds[0].empty());
  EXPECT_TRUE(dag.preds[1].empty());
}

TEST_F(DataDependenceTest, DifferentParticipants_NoEdge) {
  // Same buffer_ref, same offset, different participants -> no edge.
  Program program;
  auto src1 = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto dst1 = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  (void)program.EmitCopy(src1, dst1);
  // Second op reads shard_b on dev0, not dev1 — different participant.
  auto src2 = program.EmitView(dev0, shard_b, Slice{0, 64}, dt);
  auto dst2 = program.EmitView(dev1, shard_a, Slice{0, 64}, dt);
  (void)program.EmitCopy(src2, dst2);

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 2u);
  EXPECT_TRUE(dag.preds[0].empty());
  EXPECT_TRUE(dag.preds[1].empty());
}

TEST_F(DataDependenceTest, DifferentBufferRefs_NoEdge) {
  // Same participant, different buffer_refs -> no edge.
  Program program;
  auto src1 = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto dst1 = program.EmitView(dev1, shard_a, Slice{0, 64}, dt);
  (void)program.EmitCopy(src1, dst1);
  auto src2 = program.EmitView(dev0, shard_b, Slice{0, 64}, dt);
  auto dst2 = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  (void)program.EmitCopy(src2, dst2);

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 2u);
  EXPECT_TRUE(dag.preds[0].empty());
  EXPECT_TRUE(dag.preds[1].empty());
}

//==============================================================================
// Write supersession: older writers don't generate edges for later readers
// once a newer write has covered the same bytes.
//==============================================================================

TEST_F(DataDependenceTest, SupersededWriter_NoEdgeToOlderWrite) {
  // Each source uses a distinct shard so only the (dev1, shard_b) writes
  // and the final read participate in the dependency analysis.
  //
  // copy #0: reads (dev0, shard_a)  writes (dev1, shard_b, [0, 64))
  // copy #1: reads (dev0, shard_c)  writes (dev1, shard_b, [0, 64))  -- supersedes #0
  // copy #2: reads (dev1, shard_b, [0, 64))  writes (dev0, shard_c) -- RAW on #1 only
  Program program;
  auto a0 = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto b0 = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  (void)program.EmitCopy(a0, b0);
  auto c1 = program.EmitView(dev0, shard_c, Slice{0, 64}, dt);
  auto b1 = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  (void)program.EmitCopy(c1, b1);
  auto b2 = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  auto c2 = program.EmitView(dev0, shard_c, Slice{64, 64}, dt);
  (void)program.EmitCopy(b2, c2);

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 3u);
  // node 1 depends on node 0 (WAW).
  EXPECT_EQ(dag.preds[1], std::set<std::uint32_t>{0});
  // node 2 depends on node 1 only (RAW), not node 0 — it was superseded.
  EXPECT_EQ(dag.preds[2], std::set<std::uint32_t>{1});
}

//==============================================================================
// Pack / Unpack / AllGather
//==============================================================================

TEST_F(DataDependenceTest, Pack_OneNode_MultipleReadsAndWrites) {
  // Pack three sources into an in-memory destination view.
  // (No AllocTmp here so we avoid needing a register allocation.)
  Program program;
  auto s0 = program.EmitView(dev0, shard_a, Slice{0, 16}, dt);
  auto s1 = program.EmitView(dev0, shard_a, Slice{16, 16}, dt);
  auto s2 = program.EmitView(dev0, shard_a, Slice{32, 16}, dt);
  auto dst = program.EmitView(dev1, shard_b, Slice{0, 48}, dt);
  (void)program.EmitPack({s0, s1, s2}, dst);

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 1u);
  EXPECT_EQ(dag.nodes[0].reads.size(), 3u);
  EXPECT_EQ(dag.nodes[0].writes.size(), 3u)
      << "Pack writes one sub-range per source into the destination";
  // Destination participant is dev1; source participant is dev0.
  EXPECT_EQ(dag.nodes[0].participants.size(), 2u);
}

TEST_F(DataDependenceTest, Unpack_OneNode_MultipleReadsAndWrites) {
  Program program;
  auto src = program.EmitView(dev0, shard_a, Slice{0, 48}, dt);
  auto d0 = program.EmitView(dev1, shard_b, Slice{0, 16}, dt);
  auto d1 = program.EmitView(dev1, shard_b, Slice{16, 16}, dt);
  auto d2 = program.EmitView(dev1, shard_b, Slice{32, 16}, dt);
  (void)program.EmitUnpack(src, {d0, d1, d2});

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 1u);
  EXPECT_EQ(dag.nodes[0].reads.size(), 3u);
  EXPECT_EQ(dag.nodes[0].writes.size(), 3u);
}

TEST_F(DataDependenceTest, AllGather_OneNode_AllParticipants) {
  // Two participants, each contributing one chunk and receiving the full
  // gathered buffer.
  Program program;
  auto s0 = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev0, shard_a, Slice{0, 128}, dt);
  auto s1 = program.EmitView(dev1, shard_a, Slice{64, 64}, dt);
  auto d1 = program.EmitView(dev1, shard_a, Slice{0, 128}, dt);
  (void)program.EmitAllGather({s0, s1}, {d0, d1});

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 1u);
  EXPECT_EQ(dag.nodes[0].reads.size(), 2u);
  EXPECT_EQ(dag.nodes[0].writes.size(), 2u);
  EXPECT_EQ(dag.nodes[0].participants.size(), 2u);
}

TEST_F(DataDependenceTest, AllGatherThenCopy_EdgeToAllGatherOutput) {
  // AllGather writes dev0's shard_a [0, 128). A subsequent copy reads
  // dev0 shard_a [0, 64) -- should depend on the AllGather.
  Program program;
  auto s0 = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto d0 = program.EmitView(dev0, shard_a, Slice{0, 128}, dt);
  auto s1 = program.EmitView(dev1, shard_a, Slice{64, 64}, dt);
  auto d1 = program.EmitView(dev1, shard_a, Slice{0, 128}, dt);
  (void)program.EmitAllGather({s0, s1}, {d0, d1});

  auto r = program.EmitView(dev0, shard_a, Slice{0, 64}, dt);
  auto w = program.EmitView(dev1, shard_b, Slice{0, 64}, dt);
  (void)program.EmitCopy(r, w);

  auto dag = BuildDataDependence(program, std::nullopt);
  ASSERT_EQ(dag.nodes.size(), 2u);
  EXPECT_EQ(dag.preds[1], std::set<std::uint32_t>{0});
}

//==============================================================================
}  // namespace setu::test::native
//==============================================================================
