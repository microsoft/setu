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
#include "planner/Plan.h"
#include "planner/ir/cir/Program.h"
#include "planner/passes/PassContext.h"
//==============================================================================
namespace setu::planner::targets {
//==============================================================================

namespace cir = setu::planner::ir::cir;

/// Abstract backend that lowers a CIR Program into a per-device LLC Plan.
class Backend {
 public:
  virtual ~Backend() = default;

  /// Lower a CIR program into per-device LLC programs.
  /// The compilation context carries register sets, P2P topology, and hints.
  [[nodiscard]] virtual Plan Run(
      const cir::Program& program /*[in]*/,
      const setu::planner::passes::PassContext& ctx /*[in]*/) = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
