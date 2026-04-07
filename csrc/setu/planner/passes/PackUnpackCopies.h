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

/// Consolidates cross-device CopyOps that share the same (src_device,
/// dst_device, dtype) triple into a single pack → copy → unpack sequence.
///
/// For each group of 2+ cross-device copies between the same device pair:
///   1. Allocate a temp buffer on the source device (total size of all sources)
///   2. Pack all source values into that temp
///   3. Allocate a temp buffer on the destination device (same size)
///   4. Single cross-device copy from src temp to dst temp
///   5. Unpack dst temp into all original destination values
///
/// Same-device copies and singleton cross-device copies are left unchanged.
/// This pass should run after ShortestPathRouting.
class PackUnpackCopies : public Pass {
 public:
  PackUnpackCopies() = default;
  [[nodiscard]] cir::Program Run(cir::Program program,
                                 const PassContext& ctx) override;
  [[nodiscard]] std::string Name() const override { return "pack_unpack_copies"; }
};

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
