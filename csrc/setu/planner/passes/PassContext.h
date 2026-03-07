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
#include "planner/RegisterSet.h"
#include "planner/hints/HintStore.h"
#include "planner/ir/cir/Value.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================

/// Immutable context passed to every pass at run time.
///
/// Contains per-operation hints (routing, bandwidth) and global compiler
/// configuration (register pool sizes).  Passes read what they need and
/// ignore the rest.
struct PassContext {
  const setu::planner::hints::HintStore& hints;
  const std::unordered_map<setu::planner::ir::cir::Device,
                           setu::planner::RegisterSet>& register_sets;
};

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
