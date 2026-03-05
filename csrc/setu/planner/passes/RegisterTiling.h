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
#include "planner/Constants.h"
#include "planner/passes/Pass.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================

/// Register tiling pass that splits large AllocTmpOps into register-sized
/// chunks and adjusts associated CopyOps to operate on the smaller tiles.
///
/// For each AllocTmpOp whose size exceeds the chunk size:
///   1. Replace with ceil(N/n) register-sized AllocTmpOps.
///   2. Replace CopyOps that write to / read from the tmp with per-chunk
///      copies (slicing the non-chunked operand as needed).
///
/// The resulting program has more operations but all temp buffers fit in
/// a single physical register slot.  A subsequent InstructionScheduler pass
/// reorders operations to minimize register pressure and enable pipelining.
class RegisterTiling : public Pass {
 public:
  explicit RegisterTiling(
      std::size_t chunk_size_bytes = setu::planner::kRegisterSize)
      : chunk_size_bytes_(chunk_size_bytes) {}

  [[nodiscard]] cir::Program Run(cir::Program program,
                                 const HintStore& hints) override;
  [[nodiscard]] std::string Name() const override { return "RegisterTiling"; }

 private:
  std::size_t chunk_size_bytes_;
};

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
