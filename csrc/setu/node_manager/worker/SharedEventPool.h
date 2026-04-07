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
#include "commons/BoostCommon.h"
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/utils/CUDAUtils.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================

using setu::commons::CopyOperationId;

/// Multi-device event pool shared across all NCCLWorkers on a node.
///
/// Per-device sub-pools hold pre-allocated cudaEvent_t handles.  Events
/// are acquired by Record (SyncPoint) and released by GetEvent (Wait)
/// via atomic refcounting.
///
/// CUDA guarantee (cudaEventRecord docs): cudaStreamWaitEvent captures
/// the event state at the time of the CPU-side API call and is not
/// affected by later cudaEventRecord.  So an event is safe to reuse
/// the moment all CPU-side cudaStreamWaitEvent calls have been made —
/// no cudaEventQuery needed.
///
/// Entries are keyed by (copy_op_id, sync_id).  `Record` is
/// destructive: assigning to `active_[key]` silently replaces any
/// existing entry.  Sync ids are monotonically increasing within one
/// plan but reset to 0 across plans, so if two plans both used
/// sync_id alone as the key, a later plan's Record could overwrite
/// an earlier plan's entry before its waiters consumed it.  Scoping
/// by copy_op_id makes the key globally unique and keeps each plan's
/// entries independent.
///
/// Thread safety: per-device free lists and the active map are mutex-
/// guarded.  SyncPoint/Wait are infrequent relative to data ops, so
/// contention is negligible.
class SharedEventPool {
 public:
  explicit SharedEventPool(std::int32_t num_devices)
      : device_pools_(num_devices) {}

  ~SharedEventPool() {
    for (auto& pool : device_pools_) {
      for (auto e : pool.events) {
        cudaEventDestroy(e);
      }
    }
  }

  SharedEventPool(const SharedEventPool&) = delete;
  SharedEventPool& operator=(const SharedEventPool&) = delete;

  /// Pre-create events on the calling thread's active CUDA device.
  /// Called by each worker during Setup().
  void InitDevice(std::int32_t device_idx, std::size_t initial_size) {
    auto& pool = device_pools_[device_idx];
    std::lock_guard<std::mutex> lock(pool.mutex);
    pool.events.reserve(initial_size);
    pool.free_list.reserve(initial_size);
    for (std::size_t i = 0; i < initial_size; ++i) {
      cudaEvent_t e;
      CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      pool.events.push_back(e);
      pool.free_list.push_back(e);
    }
  }

  /// Acquire an event and record it.  Called by the worker executing
  /// a SyncPoint instruction.  `copy_op_id` scopes the sync_id to its
  /// plan so concurrent plans don't collide on the same id.
  /// `wait_count` is the number of Waits that will reference this key.
  void Record(const CopyOperationId& copy_op_id, std::uint32_t sync_id,
              cudaStream_t stream, std::int32_t device_idx,
              std::uint32_t wait_count) {
    cudaEvent_t event = AcquireFromDevice(device_idx);

    {
      std::lock_guard<std::mutex> lock(active_mutex_);
      active_[Key{copy_op_id, sync_id}] =
          ActiveEvent{event, device_idx, wait_count};
    }
    active_cv_.notify_all();

    CUDA_CHECK(cudaEventRecord(event, stream));
    LOG_DEBUG(
        "SharedEventPool::Record copy_op={} sync_id={} device={} wait_count={}",
        copy_op_id, sync_id, device_idx, wait_count);
  }

  /// Look up the event for (copy_op_id, sync_id) and decrement its
  /// refcount.  Returns the event (caller calls cudaStreamWaitEvent).
  /// When refcount hits 0, the event is returned to its device's free list.
  [[nodiscard]] cudaEvent_t GetEvent(const CopyOperationId& copy_op_id,
                                     std::uint32_t sync_id) {
    Key key{copy_op_id, sync_id};
    std::unique_lock<std::mutex> lock(active_mutex_);
    active_cv_.wait(lock, [&] { return active_.contains(key); });

    auto it = active_.find(key);
    auto& entry = it->second;
    cudaEvent_t event = entry.event;

    ASSERT_VALID_RUNTIME(
        entry.remaining > 0,
        "GetEvent: copy_op={} sync_id={} refcount already 0", copy_op_id,
        sync_id);
    entry.remaining--;

    if (entry.remaining == 0) {
      auto device_idx = entry.device_idx;
      active_.erase(it);
      ReturnToDevice(device_idx, event);
    }

    return event;
  }

 private:
  struct Key {
    CopyOperationId copy_op_id;
    std::uint32_t sync_id;
    bool operator==(const Key& other) const {
      return copy_op_id == other.copy_op_id && sync_id == other.sync_id;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
      return boost::hash<CopyOperationId>{}(k.copy_op_id) ^
             (std::hash<std::uint32_t>{}(k.sync_id) << 1);
    }
  };

  struct ActiveEvent {
    cudaEvent_t event;
    std::int32_t device_idx;
    std::uint32_t remaining;  ///< Waits not yet processed
  };

  struct DevicePool {
    std::vector<cudaEvent_t> events;     ///< All events owned by this device
    std::vector<cudaEvent_t> free_list;  ///< Available for reuse
    std::mutex mutex;
  };

  cudaEvent_t AcquireFromDevice(std::int32_t device_idx) {
    auto& pool = device_pools_[device_idx];
    std::lock_guard<std::mutex> lock(pool.mutex);

    if (!pool.free_list.empty()) {
      auto event = pool.free_list.back();
      pool.free_list.pop_back();
      return event;
    }

    // Grow: create new event on current device (caller has set device).
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    pool.events.push_back(e);
    return e;
  }

  void ReturnToDevice(std::int32_t device_idx, cudaEvent_t event) {
    auto& pool = device_pools_[device_idx];
    std::lock_guard<std::mutex> lock(pool.mutex);
    pool.free_list.push_back(event);
  }

  std::vector<DevicePool> device_pools_;
  std::unordered_map<Key, ActiveEvent, KeyHash> active_;
  std::mutex active_mutex_;
  std::condition_variable active_cv_;
};

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
