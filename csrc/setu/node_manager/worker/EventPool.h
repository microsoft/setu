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
#include <cuda_runtime.h>
//==============================================================================
#include "commons/StdCommon.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================

/// Pool of CUDA events that maps logical IDs (from SyncPoint/Wait instructions)
/// to physical cudaEvent_t handles.  Physical events are recycled via
/// scavenging: when the free list is exhausted, completed events are reclaimed
/// using cudaEventQuery.  If a scavenged event is later referenced by a Wait,
/// the wait is skipped — the dependency is already satisfied.
class EventPool {
 public:
  explicit EventPool(std::size_t initial_size = 0);
  ~EventPool();

  EventPool(const EventPool&) = delete;
  EventPool& operator=(const EventPool&) = delete;
  EventPool(EventPool&&) = delete;
  EventPool& operator=(EventPool&&) = delete;

  /// Map a logical event ID to a physical CUDA event.
  /// Recycles from free list, scavenges on pressure, or grows the pool.
  [[nodiscard]] cudaEvent_t Acquire(std::uint32_t logical_id);

  /// Get the physical event for a logical ID.
  /// Returns nullptr if the ID has been scavenged (dependency satisfied).
  [[nodiscard]] cudaEvent_t Get(std::uint32_t logical_id) const;

  /// Reset all mappings. All physical events return to the free list.
  /// Called at program boundaries.
  void Reset();

 private:
  /// Reclaim completed events via cudaEventQuery.
  void Scavenge();

  std::vector<cudaEvent_t> physical_events_;
  std::vector<std::size_t> free_list_;
  std::unordered_map<std::uint32_t, std::size_t> logical_to_physical_;
};

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
