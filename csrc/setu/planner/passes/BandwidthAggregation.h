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
#pragma once
//==============================================================================
#include "commons/StdCommon.h"
//==============================================================================
#include "planner/passes/Pass.h"
#include "planner/topo/Topology.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================
using setu::planner::topo::TopologyPtr;
//==============================================================================

/// Bandwidth aggregation pass that splits cross-device copies across multiple
/// edge-disjoint paths to aggregate link bandwidth.
///
/// For each CopyOp between different devices:
///   1. Find up to max_paths edge-disjoint paths via the topology.
///   2. Prune paths using a greedy cost model: drop paths whose added latency
///      outweighs the bandwidth benefit (handles small buffers naturally).
///   3. Split the buffer proportional to each path's bottleneck bandwidth.
///   4. Emit a multi-hop copy chain per path (allocating temp buffers at
///      intermediate hops).
class BandwidthAggregation : public Pass {
 public:
  explicit BandwidthAggregation(TopologyPtr topo, std::size_t max_paths = 4)
      : topo_(std::move(topo)), max_paths_(max_paths) {}

  [[nodiscard]] cir::Program Run(cir::Program program,
                                 const PassContext& ctx) override;
  [[nodiscard]] std::string Name() const override {
    return "bandwidth_aggregation";
  }

 private:
  TopologyPtr topo_;
  std::size_t max_paths_;
};

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
