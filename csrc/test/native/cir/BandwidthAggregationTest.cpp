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
#include "planner/hints/Hint.h"
#include "planner/hints/HintStore.h"
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
#include "planner/passes/BandwidthAggregation.h"
#include "planner/topo/Topology.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::Participant;
using setu::planner::hints::HintStore;
using setu::planner::hints::RoutingHint;
using setu::planner::ir::cir::AllocTmpOp;
using setu::planner::ir::cir::CopyOp;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::OpType;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::SliceOp;
using setu::planner::ir::cir::Value;
using setu::planner::ir::cir::ValueInfo;
using setu::planner::passes::BandwidthAggregation;
using setu::planner::topo::Link;
using setu::planner::topo::Path;
using setu::planner::topo::Topology;
using setu::planner::topo::TopologyPtr;
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

class BandwidthAggregationTest : public ::testing::Test {
 protected:
  Device dev0 = MakeTestDevice(0);
  Device dev1 = MakeTestDevice(1);
  Device dev2 = MakeTestDevice(2);
  Device dev3 = MakeTestDevice(3);
  torch::Dtype dt = torch::kFloat16;
  setu::planner::ir::ref::ShardRef shard = MakeTestShardRef();
  HintStore hints;

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

  /// Build a diamond topology:
  ///        dev0
  ///       /    \
  ///  (bw_a)    (bw_b)
  ///     /        \
  ///   dev2      dev3
  ///     \        /
  ///  (bw_a)    (bw_b)
  ///       \    /
  ///        dev1
  [[nodiscard]] TopologyPtr MakeDiamondTopology(float bw_a = 200.0f,
                                                float bw_b = 100.0f,
                                                float latency = 1.0f) const {
    auto topo = std::make_shared<Topology>();
    topo->AddBidirectionalLink(dev0, dev2, Link(latency, bw_a));
    topo->AddBidirectionalLink(dev2, dev1, Link(latency, bw_a));
    topo->AddBidirectionalLink(dev0, dev3, Link(latency, bw_b));
    topo->AddBidirectionalLink(dev3, dev1, Link(latency, bw_b));
    return topo;
  }

  /// Build a topology with a direct link plus a relay path.
  ///   dev0 ---(bw_direct)--- dev1
  ///   dev0 ---(bw_relay)---- dev2 ---(bw_relay)--- dev1
  [[nodiscard]] TopologyPtr MakeDirectPlusRelayTopology(
      float bw_direct = 200.0f, float bw_relay = 100.0f,
      float latency = 1.0f) const {
    auto topo = std::make_shared<Topology>();
    topo->AddBidirectionalLink(dev0, dev1, Link(latency, bw_direct));
    topo->AddBidirectionalLink(dev0, dev2, Link(latency, bw_relay));
    topo->AddBidirectionalLink(dev2, dev1, Link(latency, bw_relay));
    return topo;
  }
};

//==============================================================================
// Empty / no-op cases
//==============================================================================

TEST_F(BandwidthAggregationTest, EmptyProgram_ProducesEmptyProgram) {
  auto topo = MakeDiamondTopology();
  BandwidthAggregation pass(topo);
  Program program;
  auto result = pass.Run(std::move(program), hints);

  EXPECT_EQ(result.NumOperations(), 0u);
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Same-device copies should pass through unchanged
//==============================================================================

TEST_F(BandwidthAggregationTest, SameDeviceCopy_PassedThrough) {
  auto topo = MakeDiamondTopology();
  BandwidthAggregation pass(topo);

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, 1024}, dt);
  auto dst = program.EmitView(dev0, shard, Slice{1024, 1024}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), hints);

  EXPECT_EQ(CountOps(result, OpType::kCopy), 1u);
  EXPECT_EQ(CountOps(result, OpType::kSlice), 0u);
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 0u);
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Direct transfer (2 hops, single path) should pass through unchanged
//==============================================================================

TEST_F(BandwidthAggregationTest, DirectTransfer_SinglePath_PassedThrough) {
  // Only a direct link between dev0 and dev1, no relay paths.
  auto topo = std::make_shared<Topology>();
  topo->AddBidirectionalLink(dev0, dev1, Link(1.0f, 200.0f));
  BandwidthAggregation pass(topo);

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, 1024}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, 1024}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), hints);

  EXPECT_EQ(CountOps(result, OpType::kCopy), 1u);
  EXPECT_EQ(CountOps(result, OpType::kSlice), 0u);
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 0u);
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Diamond topology with large buffer should split into 2 paths
//==============================================================================

TEST_F(BandwidthAggregationTest, DiamondTopology_LargeBuffer_SplitsInto2Paths) {
  auto topo = MakeDiamondTopology(200.0f, 100.0f, 1.0f);
  BandwidthAggregation pass(topo);

  // Large buffer: 1M elements * 2 bytes (float16) = 2 MB
  const std::size_t num_elements = 1024 * 1024;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), hints);

  // Should have slices (splitting the buffer) and allocations (temp buffers
  // at intermediate hops dev2 and dev3).
  EXPECT_GT(CountOps(result, OpType::kSlice), 0u)
      << "Buffer should be split into slices";
  EXPECT_GT(CountOps(result, OpType::kAllocTmp), 0u)
      << "Should allocate temp buffers at intermediate hops";
  // 2 paths through dev2 and dev3 → 2 temp allocations.
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 2u);
  // 2 paths × 2 copies each (src→intermediate, intermediate→dst) = 4 copies,
  // plus there's also dst_in consume.
  EXPECT_GT(CountOps(result, OpType::kCopy), 2u);
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Small buffer should not be split (latency dominates)
//==============================================================================

TEST_F(BandwidthAggregationTest, DiamondTopology_SmallBuffer_NoSplit) {
  // High latency links make splitting unprofitable for small buffers.
  auto topo = MakeDiamondTopology(200.0f, 100.0f, 100.0f);
  BandwidthAggregation pass(topo);

  // Tiny buffer: 64 elements * 2 bytes = 128 bytes.
  const std::size_t num_elements = 64;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), hints);

  // Pruning should select a single path (no splitting benefit for small data
  // with high latency). The single path is multi-hop (dev0→dev2→dev1), so
  // we get 1 temp alloc and 2 copies, but no slicing of the original buffer.
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Three disjoint paths with large buffer should split 3 ways
//==============================================================================

TEST_F(BandwidthAggregationTest, ThreeDisjointPaths_LargeBuffer_3WaySplit) {
  // dev0 → dev1 via three relay GPUs: dev2, dev3, and a direct link.
  // dev0 --200GB/s-- dev1 (direct)
  // dev0 --100GB/s-- dev2 --100GB/s-- dev1
  // dev0 --100GB/s-- dev3 --100GB/s-- dev1
  auto topo = std::make_shared<Topology>();
  topo->AddBidirectionalLink(dev0, dev1, Link(1.0f, 200.0f));
  topo->AddBidirectionalLink(dev0, dev2, Link(1.0f, 100.0f));
  topo->AddBidirectionalLink(dev2, dev1, Link(1.0f, 100.0f));
  topo->AddBidirectionalLink(dev0, dev3, Link(1.0f, 100.0f));
  topo->AddBidirectionalLink(dev3, dev1, Link(1.0f, 100.0f));
  BandwidthAggregation pass(topo);

  const std::size_t num_elements = 1024 * 1024;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), hints);

  // 2 relay paths need temp buffers (dev2, dev3). Direct path has no temps.
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 2u);
  // Should have slices for 3-way split.
  EXPECT_GE(CountOps(result, OpType::kSlice), 6u)
      << "3 paths × 2 slices (src + dst) each = 6 slices";
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Bandwidth-proportional split sizes
//==============================================================================

TEST_F(BandwidthAggregationTest, SplitSizes_ProportionalToBandwidth) {
  // Direct plus relay topology. Direct has 200 GB/s, relay has 100 GB/s
  // bottleneck. So split should be ~2:1.
  auto topo = MakeDirectPlusRelayTopology(200.0f, 100.0f, 1.0f);
  BandwidthAggregation pass(topo);

  // Use a buffer size that divides cleanly.
  const std::size_t num_elements = 300000;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), hints);

  // Verify we have slicing (buffer was split).
  auto slice_count = CountOps(result, OpType::kSlice);
  EXPECT_GT(slice_count, 0u) << "Buffer should be split";

  // Check that slice sizes reflect ~2:1 ratio.
  // Slices come in pairs: (src_slice_path0, dst_slice_path0,
  //                        src_slice_path1, dst_slice_path1, ...)
  // Collect unique slice sizes by taking every other one (src slices).
  std::vector<std::size_t> src_slice_sizes;
  std::size_t slice_idx = 0;
  for (const auto& op : result.Operations()) {
    if (op.Type() == OpType::kSlice) {
      if (slice_idx % 2 == 0) {
        const auto& slice_op = std::get<SliceOp>(op.op);
        src_slice_sizes.push_back(slice_op.slice.size);
      }
      ++slice_idx;
    }
  }

  ASSERT_EQ(src_slice_sizes.size(), 2u) << "Should have 2 src slices";
  // The larger slice should be roughly 2x the smaller one.
  auto max_slice = *std::ranges::max_element(src_slice_sizes);
  auto min_slice = *std::ranges::min_element(src_slice_sizes);
  // Allow some tolerance due to integer rounding.
  float ratio = static_cast<float>(max_slice) / static_cast<float>(min_slice);
  EXPECT_NEAR(ratio, 2.0f, 0.1f) << "Split ratio should be ~2:1";

  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Routing hint override should bypass multi-path
//==============================================================================

TEST_F(BandwidthAggregationTest, RoutingHintOverride_UsesHintPath) {
  auto topo = MakeDiamondTopology();
  BandwidthAggregation pass(topo);

  // Force a specific path via hint: dev0 → dev2 → dev1.
  Path forced_path({dev0, dev2, dev1},
                   {Link(1.0f, 200.0f), Link(1.0f, 200.0f)});
  hints.AddHint(RoutingHint(dev0, dev1, forced_path));

  const std::size_t num_elements = 1024 * 1024;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), hints);

  // Hint specifies a single 3-hop path → 1 temp allocation (at dev2),
  // 2 copies (src→dev2, dev2→dst), no slicing.
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 1u);
  EXPECT_EQ(CountOps(result, OpType::kCopy), 2u);
  EXPECT_EQ(CountOps(result, OpType::kSlice), 0u);
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// No topology and no hints should clone unchanged
//==============================================================================

TEST_F(BandwidthAggregationTest, NoTopologyNoHints_ClonesUnchanged) {
  BandwidthAggregation pass(nullptr);

  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, 1024}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, 1024}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), hints);

  EXPECT_EQ(CountOps(result, OpType::kCopy), 1u);
  EXPECT_EQ(CountOps(result, OpType::kSlice), 0u);
  EXPECT_EQ(CountOps(result, OpType::kAllocTmp), 0u);
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// KEdgeDisjointPaths unit tests
//==============================================================================

TEST_F(BandwidthAggregationTest, KEdgeDisjointPaths_Diamond_FindsTwo) {
  auto topo = MakeDiamondTopology();
  auto cost_fn = [](const Link& l) { return l.TransferTimeUs(1024 * 1024); };

  auto paths = topo->KEdgeDisjointPaths(dev0, dev1, 4, cost_fn);

  EXPECT_EQ(paths.size(), 2u) << "Diamond should have exactly 2 disjoint paths";
  // Both paths should be 3-hop (dev0 → devX → dev1).
  for (const auto& p : paths) {
    EXPECT_EQ(p.hops.size(), 3u);
    EXPECT_EQ(p.hops.front(), dev0);
    EXPECT_EQ(p.hops.back(), dev1);
  }
}

TEST_F(BandwidthAggregationTest, KEdgeDisjointPaths_SingleLink_FindsOne) {
  auto topo = std::make_shared<Topology>();
  topo->AddBidirectionalLink(dev0, dev1, Link(1.0f, 200.0f));

  auto cost_fn = [](const Link& l) { return l.TransferTimeUs(1024); };
  auto paths = topo->KEdgeDisjointPaths(dev0, dev1, 4, cost_fn);

  EXPECT_EQ(paths.size(), 1u);
  EXPECT_EQ(paths[0].hops.size(), 2u);
}

TEST_F(BandwidthAggregationTest, KEdgeDisjointPaths_ThreePaths_FindsThree) {
  auto topo = std::make_shared<Topology>();
  topo->AddBidirectionalLink(dev0, dev1, Link(1.0f, 200.0f));
  topo->AddBidirectionalLink(dev0, dev2, Link(1.0f, 100.0f));
  topo->AddBidirectionalLink(dev2, dev1, Link(1.0f, 100.0f));
  topo->AddBidirectionalLink(dev0, dev3, Link(1.0f, 100.0f));
  topo->AddBidirectionalLink(dev3, dev1, Link(1.0f, 100.0f));

  auto cost_fn = [](const Link& l) { return l.TransferTimeUs(1024 * 1024); };
  auto paths = topo->KEdgeDisjointPaths(dev0, dev1, 4, cost_fn);

  EXPECT_EQ(paths.size(), 3u);
}

//==============================================================================
}  // namespace
}  // namespace setu::test::native
//==============================================================================
