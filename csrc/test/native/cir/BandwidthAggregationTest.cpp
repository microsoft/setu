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
#include "planner/hints/Hint.h"
#include "planner/hints/HintStore.h"
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
#include "planner/passes/BandwidthAggregation.h"
#include "planner/passes/PassContext.h"
#include "planner/topo/Topology.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::Participant;
using setu::planner::hints::BandwidthHint;
using setu::planner::hints::HintStore;
using setu::planner::hints::RoutingHint;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::passes::BandwidthAggregation;
using setu::planner::passes::P2PAccessMap;
using setu::planner::passes::PassContext;
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
  std::unordered_map<Device, setu::planner::RegisterSet> empty_register_sets;
  P2PAccessMap empty_p2p_access;

  PassContext DefaultCtx() {
    return PassContext{.hints = hints,
                       .register_sets = empty_register_sets,
                       .p2p_access = empty_p2p_access};
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
  auto result = pass.Run(std::move(program), DefaultCtx());

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

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [1024, 1024], Half)
  [2] %2 = copy(%0, %1)
)");
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

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [2] %2 = copy(%0, %1)
)");
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

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1048576], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1048576], Half)
  [2] %2 = slice(%0, [0, 699051])
  [3] %3 = slice(%1, [0, 699051])
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 699051, Half)
  [5] %5 = copy(%2, %4)
  [6] %6 = copy(%5, %3)
  [7] %7 = slice(%0, [699051, 349525])
  [8] %8 = slice(%1, [699051, 349525])
  [9] %9 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 349525, Half)
  [10] %10 = copy(%7, %9)
  [11] %11 = copy(%10, %8)
  [12] %12 = consume(%1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// Small buffer should not be split (latency dominates)
//==============================================================================

TEST_F(BandwidthAggregationTest, DiamondTopology_SmallBuffer_HighLatency) {
  // Small buffer with high latency links — still splits proportionally.
  auto topo = MakeDiamondTopology(200.0f, 100.0f, 100.0f);
  BandwidthAggregation pass(topo);

  // Tiny buffer: 64 elements * 2 bytes = 128 bytes.
  const std::size_t num_elements = 64;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 64], Half)
  [2] %2 = slice(%0, [0, 43])
  [3] %3 = slice(%1, [0, 43])
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 43, Half)
  [5] %5 = copy(%2, %4)
  [6] %6 = copy(%5, %3)
  [7] %7 = slice(%0, [43, 21])
  [8] %8 = slice(%1, [43, 21])
  [9] %9 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 21, Half)
  [10] %10 = copy(%7, %9)
  [11] %11 = copy(%10, %8)
  [12] %12 = consume(%1)
)");
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

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1048576], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1048576], Half)
  [2] %2 = slice(%0, [0, 524288])
  [3] %3 = slice(%1, [0, 524288])
  [4] %4 = copy(%2, %3)
  [5] %5 = slice(%0, [524288, 262144])
  [6] %6 = slice(%1, [524288, 262144])
  [7] %7 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 262144, Half)
  [8] %8 = copy(%5, %7)
  [9] %9 = copy(%8, %6)
  [10] %10 = slice(%0, [786432, 262144])
  [11] %11 = slice(%1, [786432, 262144])
  [12] %12 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 262144, Half)
  [13] %13 = copy(%10, %12)
  [14] %14 = copy(%13, %11)
  [15] %15 = consume(%1)
)");
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

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 300000], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 300000], Half)
  [2] %2 = slice(%0, [0, 200000])
  [3] %3 = slice(%1, [0, 200000])
  [4] %4 = copy(%2, %3)
  [5] %5 = slice(%0, [200000, 100000])
  [6] %6 = slice(%1, [200000, 100000])
  [7] %7 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 100000, Half)
  [8] %8 = copy(%5, %7)
  [9] %9 = copy(%8, %6)
  [10] %10 = consume(%1)
)");
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

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1048576], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1048576], Half)
  [2] %2 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 1048576, Half)
  [3] %3 = copy(%0, %2)
  [4] %4 = copy(%3, %1)
)");
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

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1024], Half)
  [2] %2 = copy(%0, %1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

//==============================================================================
// BandwidthHint: explicit multi-path splitting
//==============================================================================

TEST_F(BandwidthAggregationTest, BandwidthHint_SinglePath_EmitsCopyChain) {
  BandwidthAggregation pass(nullptr);

  // Single multi-hop path via hint, weight [1.0].
  Path path({dev0, dev2, dev1}, {Link(1.0f, 200.0f), Link(1.0f, 200.0f)});
  hints.AddHint(BandwidthHint(dev0, dev1, {path}, {1.0f}));

  const std::size_t num_elements = 1024 * 1024;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1048576], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1048576], Half)
  [2] %2 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 1048576, Half)
  [3] %3 = copy(%0, %2)
  [4] %4 = copy(%3, %1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(BandwidthAggregationTest, BandwidthHint_TwoPaths_EqualWeights) {
  BandwidthAggregation pass(nullptr);

  Path path_a({dev0, dev2, dev1}, {Link(1.0f, 200.0f), Link(1.0f, 200.0f)});
  Path path_b({dev0, dev3, dev1}, {Link(1.0f, 100.0f), Link(1.0f, 100.0f)});
  hints.AddHint(BandwidthHint(dev0, dev1, {path_a, path_b}, {0.5f, 0.5f}));

  const std::size_t num_elements = 1000;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1000], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1000], Half)
  [2] %2 = slice(%0, [0, 500])
  [3] %3 = slice(%1, [0, 500])
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 500, Half)
  [5] %5 = copy(%2, %4)
  [6] %6 = copy(%5, %3)
  [7] %7 = slice(%0, [500, 500])
  [8] %8 = slice(%1, [500, 500])
  [9] %9 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 500, Half)
  [10] %10 = copy(%7, %9)
  [11] %11 = copy(%10, %8)
  [12] %12 = consume(%1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(BandwidthAggregationTest, BandwidthHint_ThreePaths_UnequalWeights) {
  BandwidthAggregation pass(nullptr);

  Path path_a({dev0, dev1}, {Link(1.0f, 200.0f)});  // direct
  Path path_b({dev0, dev2, dev1}, {Link(1.0f, 100.0f), Link(1.0f, 100.0f)});
  Path path_c({dev0, dev3, dev1}, {Link(1.0f, 100.0f), Link(1.0f, 100.0f)});
  hints.AddHint(
      BandwidthHint(dev0, dev1, {path_a, path_b, path_c}, {0.5f, 0.3f, 0.2f}));

  const std::size_t num_elements = 10000;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 10000], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 10000], Half)
  [2] %2 = slice(%0, [0, 5000])
  [3] %3 = slice(%1, [0, 5000])
  [4] %4 = copy(%2, %3)
  [5] %5 = slice(%0, [5000, 3000])
  [6] %6 = slice(%1, [5000, 3000])
  [7] %7 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 3000, Half)
  [8] %8 = copy(%5, %7)
  [9] %9 = copy(%8, %6)
  [10] %10 = slice(%0, [8000, 2000])
  [11] %11 = slice(%1, [8000, 2000])
  [12] %12 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 2000, Half)
  [13] %13 = copy(%10, %12)
  [14] %14 = copy(%13, %11)
  [15] %15 = consume(%1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(BandwidthAggregationTest, BandwidthHint_OverridesTopology) {
  // Topology would discover diamond paths, but hint forces a single path.
  auto topo = MakeDiamondTopology();
  BandwidthAggregation pass(topo);

  Path forced_path({dev0, dev2, dev1},
                   {Link(1.0f, 200.0f), Link(1.0f, 200.0f)});
  hints.AddHint(BandwidthHint(dev0, dev1, {forced_path}, {1.0f}));

  const std::size_t num_elements = 1024 * 1024;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1048576], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1048576], Half)
  [2] %2 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 1048576, Half)
  [3] %3 = copy(%0, %2)
  [4] %4 = copy(%3, %1)
)");
  EXPECT_NO_THROW(Linearity::Check(result));
}

TEST_F(BandwidthAggregationTest, BandwidthHint_OverridesRoutingHint) {
  BandwidthAggregation pass(nullptr);

  // RoutingHint: single path through dev2.
  Path routing_path({dev0, dev2, dev1},
                    {Link(1.0f, 200.0f), Link(1.0f, 200.0f)});
  hints.AddHint(RoutingHint(dev0, dev1, routing_path));

  // BandwidthHint: two paths (should win).
  Path path_a({dev0, dev2, dev1}, {Link(1.0f, 200.0f), Link(1.0f, 200.0f)});
  Path path_b({dev0, dev3, dev1}, {Link(1.0f, 100.0f), Link(1.0f, 100.0f)});
  hints.AddHint(BandwidthHint(dev0, dev1, {path_a, path_b}, {0.5f, 0.5f}));

  const std::size_t num_elements = 1000;
  Program program;
  auto src = program.EmitView(dev0, shard, Slice{0, num_elements}, dt);
  auto dst = program.EmitView(dev1, shard, Slice{0, num_elements}, dt);
  (void)program.EmitCopy(src, dst);

  auto result = pass.Run(std::move(program), DefaultCtx());

  EXPECT_EQ(result.Dump(), R"(
  [0] %0 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1000], Half)
  [1] %1 = view(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1)), &ShardRef(shard_id=00000000-0000-0000-0000-000000000000, tensor_name=<none>, node_id=<none>), [0, 1000], Half)
  [2] %2 = slice(%0, [0, 500])
  [3] %3 = slice(%1, [0, 500])
  [4] %4 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:2)), 500, Half)
  [5] %5 = copy(%2, %4)
  [6] %6 = copy(%5, %3)
  [7] %7 = slice(%0, [500, 500])
  [8] %8 = slice(%1, [500, 500])
  [9] %9 = alloc_tmp(Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:3)), 500, Half)
  [10] %10 = copy(%7, %9)
  [11] %11 = copy(%10, %8)
  [12] %12 = consume(%1)
)");
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
