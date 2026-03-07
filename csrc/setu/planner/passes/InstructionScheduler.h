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

/// Instruction scheduler that reorders CIR operations to minimize live
/// AllocTmp register pressure while preserving data dependencies.
///
/// Uses a priority-based topological sort with a pressure guard:
///   - At each step, pick the ready op (all dependencies scheduled) with
///     the highest score.
///   - score = -1 for AllocTmpOp (penalizes starting new work), 0 otherwise.
///   - Tie-break: original program order (stability).
///   - Pressure guard: when register_sets are provided in the PassContext,
///     AllocTmpOps are deferred when scheduling them would exceed the
///     device's register pool size. Deferred ops are re-enqueued when a
///     register is freed (all alias-chain uses exhausted).
///
/// When no register_sets are available, the scheduler runs without the
/// pressure guard (score-based interleaving only).
class InstructionScheduler : public Pass {
 public:
  InstructionScheduler() = default;

  [[nodiscard]] cir::Program Run(cir::Program program,
                                 const PassContext& ctx) override;
  [[nodiscard]] std::string Name() const override {
    return "InstructionScheduler";
  }
};

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
