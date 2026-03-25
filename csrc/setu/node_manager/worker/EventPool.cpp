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
#include "node_manager/worker/EventPool.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/CUDAUtils.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================

EventPool::EventPool(std::size_t initial_size) {
  physical_events_.reserve(initial_size);
  free_list_.reserve(initial_size);
  for (std::size_t i = 0; i < initial_size; ++i) {
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    physical_events_.push_back(e);
    free_list_.push_back(i);
  }
}

EventPool::~EventPool() {
  for (auto e : physical_events_) {
    cudaEventDestroy(e);
  }
}

cudaEvent_t EventPool::Acquire(std::uint32_t logical_id) {
  std::size_t phys_idx;

  if (!free_list_.empty()) {
    phys_idx = free_list_.back();
    free_list_.pop_back();
  } else {
    Scavenge();
    if (!free_list_.empty()) {
      phys_idx = free_list_.back();
      free_list_.pop_back();
    } else {
      // Grow the pool.
      cudaEvent_t e;
      CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      phys_idx = physical_events_.size();
      physical_events_.push_back(e);
    }
  }

  logical_to_physical_[logical_id] = phys_idx;
  return physical_events_[phys_idx];
}

cudaEvent_t EventPool::Get(std::uint32_t logical_id) const {
  auto it = logical_to_physical_.find(logical_id);
  if (it == logical_to_physical_.end()) {
    return nullptr;  // Scavenged — dependency already satisfied.
  }
  return physical_events_[it->second];
}

void EventPool::Reset() {
  logical_to_physical_.clear();
  free_list_.clear();
  for (std::size_t i = 0; i < physical_events_.size(); ++i) {
    free_list_.push_back(i);
  }
}

void EventPool::Scavenge() {
  std::vector<std::uint32_t> to_free;
  for (const auto& [logical_id, phys_idx] : logical_to_physical_) {
    cudaError_t status = cudaEventQuery(physical_events_[phys_idx]);
    if (status == cudaSuccess) {
      free_list_.push_back(phys_idx);
      to_free.push_back(logical_id);
    }
    // cudaErrorNotReady means still in-flight — leave it.
  }
  for (auto id : to_free) {
    logical_to_physical_.erase(id);
  }
  LOG_DEBUG("EventPool::Scavenge: reclaimed {} events, {} still active",
            to_free.size(), logical_to_physical_.size());
}

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
