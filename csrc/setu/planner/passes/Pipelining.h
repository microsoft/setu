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
//==============================================================================
namespace setu::planner::passes {
//==============================================================================

/// Pipelining pass that splits multi-hop relay copy chains into chunks and
/// emits them in wavefront order for overlapped execution.
///
/// A relay chain is a sequence of CopyOps where each copy's dst_out feeds
/// into the next copy's src (possibly through Slice/Consume).  The pass:
///   1. Detects all maximal linear relay chains (asserts no branching).
///   2. For chains with payload > chunk_size_bytes / element_size, splits into
///      ceil(payload / chunk_size_elements) chunks.
///   3. Emits chunks in wavefront order: micro_stage = chunk_idx + hop_idx.
///      This lets chunk N's later hops overlap with chunk N+1's earlier hops.
///
/// Single-hop copies and small payloads are passed through unchanged.
///
/// The chunk size can be overridden per-operation via PipelineChunkSizeHint
/// in the PassContext.
class Pipelining : public Pass {
 public:
  explicit Pipelining(std::size_t chunk_size_bytes)
      : chunk_size_bytes_(chunk_size_bytes) {}

  [[nodiscard]] cir::Program Run(cir::Program program,
                                 const PassContext& ctx) override;
  [[nodiscard]] std::string Name() const override { return "Pipelining"; }

 private:
  std::size_t chunk_size_bytes_;
};

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
