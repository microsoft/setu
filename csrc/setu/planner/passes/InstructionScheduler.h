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
/// Uses a priority-based topological sort:
///   - At each step, pick the ready op (all dependencies scheduled) with
///     the highest score.
///   - score = (# AllocTmp-origin values freed) - (1 if AllocTmpOp, else 0)
///   - Tie-break: original program order (stability).
///
/// This naturally interleaves alloc/write/read sequences produced by
/// RegisterTiling, keeping register pressure bounded regardless of how
/// many chunks exist.
class InstructionScheduler : public Pass {
 public:
  InstructionScheduler() = default;

  [[nodiscard]] cir::Program Run(cir::Program program,
                                 const HintStore& hints) override;
  [[nodiscard]] std::string Name() const override {
    return "InstructionScheduler";
  }
};

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
