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

/// Instruction scheduler that reorders CIR operations only when necessary
/// to keep register pressure within budget.
///
/// If the program's original order already fits within the register pool
/// (checked via pressure simulation), the program is returned unchanged.
/// This preserves upstream ordering such as wavefront order from the
/// Pipelining pass.
///
/// When reordering IS needed, uses a priority-based topological sort:
///   - score = -1 for AllocTmpOp (defer new allocations), 0 otherwise.
///   - Tie-break: original program order.
///   - Pressure guard (when register_sets provided): defers AllocTmpOps
///     that would exceed the device's register pool.
class InstructionScheduler : public Pass {
 public:
  InstructionScheduler() = default;

  [[nodiscard]] cir::Program Run(cir::Program program,
                                 const PassContext& ctx) override;
  [[nodiscard]] std::string Name() const override {
    return "instruction_scheduler";
  }
};

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
