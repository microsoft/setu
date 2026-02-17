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
#include <nccl.h>
//==============================================================================
#include "planner/targets/backend.h"
#include "planner/Planner.h"
#include "planner/ir/cir/Program.h"
//==============================================================================
namespace setu::planner::targets {
//==============================================================================

using setu::planner::targets::Backend;
using setu::commons::DeviceRank;
namespace cir = setu::planner::ir::cir;

/// CIR → LLC lowering pass (NCCL flavor).
///
/// Walks a CIR Program and produces a Plan containing per-device LLC programs.
/// Maintains a communicator cache across invocations so that repeated calls
/// with the same participant set reuse the existing communicator (UseComm)
/// instead of creating a new one (InitComm).
///
/// Currently supported CIR operations:
///   view  — records shard/offset metadata for later use by copy
///   copy  — emits LLC Copy (same-device) or Send+Receive (cross-device)
///
/// Unsupported (will assert): alloc_tmp, pack, unpack
struct NCCL : public Backend {
  [[nodiscard]] Plan Run(const cir::Program& program /*[in]*/) override;

 private:
  struct CommCacheEntry {
    ncclUniqueId id;
    std::unordered_map<Participant, DeviceRank> ranks;
  };
  std::map<Participants, CommCacheEntry> comm_cache_;
};

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
