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
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/DeviceId.h"
//==============================================================================
#include "planner/Participant.h"
#include "planner/ir/llc/CommId.h"
#include "planner/ir/ref/BufferRef.h"
//==============================================================================
namespace setu::planner::targets {
//==============================================================================

/// Source of fresh ncclUniqueIds. Defaults to `ncclGetUniqueId` in
/// production; tests pass a deterministic counter-based generator so
/// that LLC outputs are byte-for-byte reproducible across runs.
using UniqueIdGenerator = std::function<ncclUniqueId()>;

/// One cached NCCL communicator: its id plus the rank assignment used
/// to create it (deterministic, derived from the participant set).
struct CommCacheEntry {
  setu::planner::ir::llc::CommId id;
  std::unordered_map<Participant, setu::commons::DeviceRank> ranks;
};

/// Map keyed by the participant set that owns the comm.
using CommCache = std::map<Participants, CommCacheEntry>;

/// Which LLC instruction a batched memcpy entry lowers to.
enum class BatchKind { kCopy, kPull };

/// A single batched one-sided memcpy entry accumulated on a participant's
/// stream. Same-participant copies and P2P pulls share this shape; `kind`
/// decides whether the run emits as `llc::Copy` or `llc::Pull` at flush
/// time. `src_device` is only meaningful for `kPull` entries.
struct BatchEntry {
  BatchKind kind;
  setu::planner::ir::ref::BufferRef src_ref;
  std::size_t src_offset_bytes;
  setu::planner::ir::ref::BufferRef dst_ref;
  std::size_t dst_offset_bytes;
  std::size_t count;
  torch::Dtype dtype;
  setu::commons::datatypes::DeviceId src_device;
};

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
