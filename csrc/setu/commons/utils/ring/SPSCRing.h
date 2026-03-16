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
#include "commons/Logging.h"
//==============================================================================
namespace setu::commons::utils::ring {
//==============================================================================

/// @brief Header layout for a lock-free SPSC ring buffer, placed at the start
/// of a contiguous memory region (heap, mmap, etc.).
///
/// `head` and `tail` are cache-line aligned to prevent false sharing.
template <typename T>
struct SPSCRingHeader {
  alignas(64) std::atomic<std::uint64_t> head;  // producer writes
  alignas(64) std::atomic<std::uint64_t> tail;  // consumer writes
  std::uint32_t capacity;                       // power-of-2, immutable
  std::uint32_t mask;                           // capacity - 1

  /// @brief Returns a pointer to the data region immediately after the header.
  [[nodiscard]] T* Data() {
    return reinterpret_cast<T*>(reinterpret_cast<std::uint8_t*>(this) +
                                sizeof(SPSCRingHeader<T>));
  }

  [[nodiscard]] const T* Data() const {
    return reinterpret_cast<const T*>(
        reinterpret_cast<const std::uint8_t*>(this) +
        sizeof(SPSCRingHeader<T>));
  }
};

//==============================================================================

/// @brief Producer end of a lock-free SPSC ring buffer.
///
/// Writes entries at `head & mask`, stores `head+1` with release ordering.
/// Spin-yields if full, assert-crashes after generous retries.
template <typename T>
class SPSCRingProducer {
 public:
  /// @brief Construct a producer from a pre-initialized memory region.
  /// @param region Pointer to the start of the memory region containing the
  ///              SPSCRingHeader followed by capacity * sizeof(T) bytes.
  explicit SPSCRingProducer(void* region)
      : header_(static_cast<SPSCRingHeader<T>*>(region)) {
    ASSERT_VALID_POINTER_ARGUMENT(region);
  }

  /// @brief Push an entry into the ring. Spins if full.
  void Push(const T& entry) {
    const auto head = header_->head.load(std::memory_order_relaxed);
    const auto capacity = header_->capacity;

    // Spin-wait until there is space
    static constexpr std::uint64_t kMaxSpinRetries = 1'000'000'000ULL;
    std::uint64_t retries = 0;
    while (head - header_->tail.load(std::memory_order_acquire) >= capacity) {
      ++retries;
      ASSERT_VALID_RUNTIME(retries < kMaxSpinRetries,
                           "SPSCRingProducer::Push: ring full after {} spins, "
                           "head={}, tail={}, capacity={}",
                           retries, head,
                           header_->tail.load(std::memory_order_relaxed),
                           capacity);
      std::this_thread::yield();
    }

    header_->Data()[head & header_->mask] = entry;
    header_->head.store(head + 1, std::memory_order_release);
  }

 private:
  SPSCRingHeader<T>* header_;
};

//==============================================================================

/// @brief Consumer end of a lock-free SPSC ring buffer.
///
/// Reads entries from `tail` up to `head`, stores updated `tail` with release.
template <typename T>
class SPSCRingConsumer {
 public:
  /// @brief Construct a consumer from a pre-initialized memory region.
  explicit SPSCRingConsumer(void* region)
      : header_(static_cast<SPSCRingHeader<T>*>(region)) {
    ASSERT_VALID_POINTER_ARGUMENT(region);
  }

  /// @brief Non-blocking poll: drains up to `max_batch` entries into `out`.
  /// @param out Output vector — entries are appended (not cleared).
  /// @param max_batch Maximum number of entries to drain in one call.
  /// @return Number of entries drained.
  [[nodiscard]] std::uint32_t Poll(std::vector<T>& out,
                                   std::uint32_t max_batch) {
    const auto head = header_->head.load(std::memory_order_acquire);
    auto tail = header_->tail.load(std::memory_order_relaxed);
    const auto mask = header_->mask;
    const T* data = header_->Data();

    std::uint32_t count = 0;
    while (tail != head && count < max_batch) {
      out.push_back(data[tail & mask]);
      ++tail;
      ++count;
    }

    if (count > 0) {
      header_->tail.store(tail, std::memory_order_release);
    }

    return count;
  }

 private:
  SPSCRingHeader<T>* header_;
};

//==============================================================================
}  // namespace setu::commons::utils::ring
//==============================================================================
