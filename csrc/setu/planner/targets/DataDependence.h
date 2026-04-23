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
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/Participant.h"
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
#include "planner/ir/ref/BufferRef.h"
//==============================================================================
namespace setu::planner::targets {
//==============================================================================

/// A contiguous byte range on one participant's buffer. Unit of
/// dependency tracking in the data-dependence graph: two operations on
/// the same (participant, buffer_ref) whose ranges overlap induce an
/// edge.
struct Region {
  setu::planner::Participant participant;
  setu::planner::ir::ref::BufferRef buffer_ref;
  std::size_t start_bytes;  ///< inclusive
  std::size_t end_bytes;    ///< exclusive
  torch::Dtype dtype;
};

/// One node in the data-dependence graph. Wraps a CIR data-moving op
/// (CopyOp, PackOp, UnpackOp, AllGatherOp) and the memory regions it
/// reads and writes. View / AllocTmp / Slice / Consume are reference
/// builders and do not become nodes.
struct DataDependenceNode {
  std::uint32_t op_idx;  ///< index into the CIR program's Operations()
  std::vector<Region> reads;
  std::vector<Region> writes;
  std::set<setu::planner::Participant> participants;
};

/// Data-dependence graph for one CIR program.
///
/// `nodes` are in CIR program order (which is itself a valid
/// topological order). `preds[i]` is the set of node indices that
/// node i depends on, derived from buffer-overlap analysis: for each
/// read region, edges from prior writes overlapping it; for each
/// write region, edges from prior reads or writes overlapping it
/// (RAW / WAW / WAR). `succs[i]` is the transpose of `preds`: the
/// set of nodes that depend on node i. Both are maintained by the
/// builder in a single pass; users needing forward adjacency (for
/// example a Kahn frontier walk) can read it directly.
struct DataDependence {
  std::vector<DataDependenceNode> nodes;
  std::vector<std::set<std::uint32_t>> preds;  // indexed by node_idx
  std::vector<std::set<std::uint32_t>> succs;  // indexed by node_idx
};

/// Build the data-dependence graph by walking the CIR program once.
///
/// `reg_alloc` must be provided iff the program contains any
/// AllocTmpOp. Callers can match what the NCCL backend already does
/// (`nccl.cpp:92-103`).
[[nodiscard]] DataDependence BuildDataDependence(
    const setu::planner::ir::cir::Program& program,
    const std::optional<setu::planner::ir::cir::RegisterAllocation>&
        reg_alloc);

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
