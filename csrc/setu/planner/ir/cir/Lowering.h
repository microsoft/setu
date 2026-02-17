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
#include "commons/datatypes/CopySpec.h"
#include "metastore/MetaStore.h"
#include "planner/ir/cir/Program.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================

using setu::commons::datatypes::CopySpec;
using setu::metastore::MetaStore;

/// Lowers a CopySpec into a CIR Program.
///
/// Uses a two-pointer walk over source and destination selections to match
/// buffer regions. For each matched region, emits:
///   - view() for the src shard slice (element offsets)
///   - view() for the dst shard slice (element offsets)
///   - copy() from src view to dst view
class CIRLowering {
 public:
  [[nodiscard]] static Program Lower(CopySpec& copy_spec /*[in]*/,
                                     MetaStore& metastore /*[in]*/);
};

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================
