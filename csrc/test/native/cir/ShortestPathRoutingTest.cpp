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
#include "commons/Types.h"
#include "planner/hints/HintStore.h"
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
#include "planner/passes/ShortestPathRouting.h"
#include "planner/topo/Topology.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::commons::NodeId;
using setu::planner::Participant;
using setu::planner::hints::HintStore;
using setu::planner::hints::RoutingHint;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::Value;
using setu::planner::passes::ShortestPathRouting;
using setu::planner::topo::Link;
using setu::planner::topo::Path;
using setu::planner::topo::Topology;

Device MakeDevice(NodeId node_id, std::int16_t gpu_index = 0) {
  return Device(node_id, setu::commons::datatypes::Device(torch::Device(
                             torch::kCUDA, static_cast<int8_t>(gpu_index))));
}

NodeId MakeNodeId(std::string uuid_str) {
  boost::uuids::string_generator gen;
  return gen(uuid_str);
}

setu::planner::ir::ref::ShardRef MakeEmptyShardRef() {
  return setu::planner::ir::ref::ShardRef(boost::uuids::nil_uuid());
}

//==============================================================================
namespace {

class CIRShortestPathRoutingTest : public testing::Test {
 protected:
  CIRShortestPathRoutingTest() {}

  NodeId n0 = MakeNodeId("01234567-89ab-cdef-0123-456789abcdef");
  NodeId n1 = MakeNodeId("00234567-89ab-cdef-0123-456789abcdef");
  torch::Dtype dt = torch::kFloat16;
};

TEST_F(CIRShortestPathRoutingTest, Scratch) {
  Program program;
  auto dev0 = MakeDevice(n0, 0);
  auto dev1 = MakeDevice(n0, 1);
  auto dev2 = MakeDevice(n1, 0);
  auto dev3 = MakeDevice(n1, 1);

  auto topo = std::make_shared<Topology>();
  topo->AddBidirectionalLink(dev0, dev1, Link(0, 200));
  topo->AddBidirectionalLink(dev0, dev2, Link(10, 100));
  topo->AddBidirectionalLink(dev0, dev3, Link(10, 50));
  topo->AddBidirectionalLink(dev1, dev2, Link(10, 50));
  topo->AddBidirectionalLink(dev1, dev3, Link(10, 100));

  auto v0 = program.EmitView(dev0, MakeEmptyShardRef(), Slice{0, 256}, dt);
  auto v1 = program.EmitView(dev0, MakeEmptyShardRef(), Slice{256, 256}, dt);
  auto v2 = program.EmitView(dev2, MakeEmptyShardRef(), Slice{0, 256}, dt);
  auto v3 = program.EmitView(dev3, MakeEmptyShardRef(), Slice{256, 256}, dt);
  (void)program.EmitCopy(v0, v2);
  (void)program.EmitCopy(v1, v3);

  std::cout << "BEFORE" << std::endl;
  std::cout << program.Dump() << std::endl;

  ShortestPathRouting pass(topo);
  HintStore hints;
  auto program_t = pass.Run(program, hints);

  std::cout << "AFTER" << std::endl;
  std::cout << program_t.Dump() << std::endl;

  EXPECT_EQ(true, false);
}

TEST_F(CIRShortestPathRoutingTest, RoutingHintOverridesPath) {
  Program program;
  auto dev0 = MakeDevice(n0, 0);
  auto dev1 = MakeDevice(n0, 1);
  auto dev2 = MakeDevice(n1, 0);

  auto topo = std::make_shared<Topology>();
  topo->AddBidirectionalLink(dev0, dev1, Link(0, 200));
  topo->AddBidirectionalLink(dev0, dev2, Link(10, 100));
  topo->AddBidirectionalLink(dev1, dev2, Link(10, 50));

  // Direct path dev0->dev2 exists, but provide a hint to route via dev1
  auto v0 = program.EmitView(dev0, MakeEmptyShardRef(), Slice{0, 256}, dt);
  auto v2 = program.EmitView(dev2, MakeEmptyShardRef(), Slice{0, 256}, dt);
  (void)program.EmitCopy(v0, v2);

  // Create a routing hint: dev0 -> dev1 -> dev2
  Path forced_path({Participant(n0, dev0.device), Participant(n0, dev1.device),
                    Participant(n1, dev2.device)},
                   {Link(0, 200), Link(10, 50)});

  HintStore hints;
  hints.AddHint(RoutingHint(Participant(n0, dev0.device),
                            Participant(n1, dev2.device), forced_path));

  ShortestPathRouting pass(topo);

  // Without hints — should produce direct copy (2 hops, no intermediates)
  HintStore empty_hints;
  auto program_no_hint = pass.Run(program, empty_hints);
  std::cout << "WITHOUT HINT" << std::endl;
  std::cout << program_no_hint.Dump() << std::endl;

  // With hints — should produce multi-hop copy (3 hops, 1 intermediate)
  auto program_with_hint = pass.Run(program, hints);
  std::cout << "WITH HINT" << std::endl;
  std::cout << program_with_hint.Dump() << std::endl;

  // The hinted program should have more operations due to intermediate hops
  EXPECT_GT(program_with_hint.NumOperations(), program_no_hint.NumOperations());
}

}  // namespace
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
