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
namespace setu::planner::passes {
//==============================================================================

using setu::commons::datatypes::CopySpec;
using setu::metastore::MetaStore;
namespace cir = setu::planner::ir::cir;

/// CopySpec → CIR lowering pass.
///
/// Walks the source and destination TensorOwnershipMaps with a two-pointer
/// algorithm, matching buffer regions in row-major order.  For each matched
/// chunk it emits:
///   %src = view(src_device, &src_shard, [offset, size], dtype)
///   %dst = view(dst_device, &dst_shard, [offset, size], dtype)
///   %dst' = copy(%src, %dst)
struct CopySpecToCIR {
  [[nodiscard]] static cir::Program Run(const CopySpec& copy_spec /*[in]*/,
                                        MetaStore& metastore /*[in]*/);
};

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
