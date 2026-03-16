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
#include "commons/ClassTraits.h"
#include "commons/utils/ring/SPSCRing.h"
//==============================================================================
namespace setu::commons::utils::ring {
//==============================================================================

/// @brief Static utility class for shared-memory lifecycle management of
/// SPSC ring buffers.
class ShmRing : public StaticClass {
 public:
  /// @brief Create a new shared-memory region and initialize the ring header.
  /// @tparam T Entry type for the ring.
  /// @param shm_name POSIX shared-memory name (must start with '/').
  /// @param capacity Number of entries (must be a power of 2).
  /// @return Pointer to the mmap'd region.
  template <typename T>
  [[nodiscard]] static void* Create(const std::string& shm_name,
                                    std::uint32_t capacity) {
    const auto total_size = ComputeSize<T>(capacity);
    void* ptr = CreateRaw(shm_name, total_size);

    // Initialize the header
    auto* header = static_cast<SPSCRingHeader<T>*>(ptr);
    new (&header->head) std::atomic<std::uint64_t>(0);
    new (&header->tail) std::atomic<std::uint64_t>(0);
    header->capacity = capacity;
    header->mask = capacity - 1;

    return ptr;
  }

  /// @brief Open an existing shared-memory region (no initialization).
  /// @tparam T Entry type for the ring.
  /// @param shm_name POSIX shared-memory name.
  /// @param capacity Number of entries (must match the region that was
  /// created).
  /// @return Pointer to the mmap'd region.
  template <typename T>
  [[nodiscard]] static void* Open(const std::string& shm_name,
                                  std::uint32_t capacity) {
    const auto total_size = ComputeSize<T>(capacity);
    return OpenRaw(shm_name, total_size);
  }

  /// @brief Unmap and unlink a shared-memory region.
  static void Destroy(const std::string& shm_name, void* ptr, std::size_t size);

  /// @brief Compute the total mmap size for a ring of the given type.
  template <typename T>
  [[nodiscard]] static std::size_t ComputeSize(std::uint32_t capacity) {
    return sizeof(SPSCRingHeader<T>) + capacity * sizeof(T);
  }

  /// @brief Generate a deterministic SHM name from a prefix and identity.
  [[nodiscard]] static std::string GenerateShmName(const std::string& prefix,
                                                   const std::string& identity);

  /// @brief Round up to the next power of 2 (no-op if already a power of 2).
  [[nodiscard]] static std::uint32_t NextPowerOf2(std::uint32_t v);

 private:
  /// Raw shm_open + mmap helpers (not templated).
  [[nodiscard]] static void* CreateRaw(const std::string& shm_name,
                                       std::size_t total_size);
  [[nodiscard]] static void* OpenRaw(const std::string& shm_name,
                                     std::size_t total_size);
};

//==============================================================================
}  // namespace setu::commons::utils::ring
//==============================================================================
