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
#include "commons/datatypes/DeviceId.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::datatypes::DeviceId;
//==============================================================================
/**
 * @brief Bidirectional map between local device indices and DeviceIds.
 *
 * The map stores `int <-> DeviceId` entries. Process-local instances are
 * obtained via DeviceMap::Local(); the singleton is populated lazily on
 * first access. The class itself is torch- and device-class-agnostic;
 * the lazy populator (private to the .cpp) chooses the right backend
 * (NVML for NVIDIA GPUs today; AMD/TPU/etc. backends will be added as
 * they are needed).
 */
class DeviceMap {
 public:
  DeviceMap() = default;

  /// Insert a (local_index, id) entry. Both directions are populated.
  void Insert(int local_index /*[in]*/, DeviceId id /*[in]*/);

  /// Returns the DeviceId for a given local index. ASSERT-fails on miss.
  [[nodiscard]] DeviceId GetDeviceId(int local_index /*[in]*/) const;

  /// Returns the local device index for a given DeviceId. ASSERT-fails
  /// on miss (the physical device is not visible to this process).
  [[nodiscard]] int GetLocalIndex(const DeviceId& id /*[in]*/) const;

  /// Number of entries.
  [[nodiscard]] std::size_t Size() const { return index_to_id_.size(); }

  /// Whether a given local index has an entry.
  [[nodiscard]] bool Contains(int local_index /*[in]*/) const {
    return index_to_id_.find(local_index) != index_to_id_.end();
  }

  /// Whether a given DeviceId has an entry.
  [[nodiscard]] bool Contains(const DeviceId& id /*[in]*/) const {
    return id_to_index_.find(id) != id_to_index_.end();
  }

  /**
   * @brief Process-local DeviceMap, populated on first access.
   *
   * Today this is built by scanning visible NVIDIA GPUs via NVML; over
   * time the populator will learn about additional device classes.
   */
  [[nodiscard]] static const DeviceMap& Local();

 private:
  std::unordered_map<int, DeviceId> index_to_id_;
  std::unordered_map<DeviceId, int> id_to_index_;
};
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
