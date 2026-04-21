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
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/DeviceId.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================
/**
 * @brief Represents a physical compute device.
 *
 * Device pairs a process-local torch::Device with a canonical, process-
 * invariant DeviceId. The DeviceId is what travels across process
 * boundaries; on construction (locally) and on deserialization (from the
 * wire) the two fields are kept in sync via DeviceMap::Local() so that
 * `LocalDeviceIndex()` is always valid in the current process.
 */
struct Device {
  /**
   * @brief Default constructor for an empty (sentinel) device.
   */
  Device() = default;

  /**
   * @brief Constructs a Device from a local torch device.
   *
   * Fills `device_id` from DeviceMap::Local() when possible.
   */
  explicit Device(torch::Device torch_device_param /*[in]*/);

  /**
   * @brief Constructs a Device from a canonical DeviceId.
   *
   * Fills `torch_device` from DeviceMap::Local() when possible.
   */
  explicit Device(DeviceId device_id_param /*[in]*/);

  // TODO: add an Available() method to help gate whether a Device is fully
  // initialized and available for use in this process.

  /**
   * @brief Returns a string representation of the device.
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("Device(torch_device={}, device_id={})",
                       torch_device.str(), device_id.ToString());
  }

  void Serialize(BinaryBuffer& buffer) const;

  static Device Deserialize(const BinaryRange& range);

  // Equality / ordering / hashing key on device_id alone.
  //
  // device_id (for example, NVML UUID for CUDA) is the canonical,
  // process-invariant identity; it uniquely identifies a physical device
  // within a node. torch_device.index() is a process-local index derived
  // from CUDA_VISIBLE_DEVICES ordering, so the same cuda:N can refer to
  // different physical devices in different processes, and is deliberately
  // excluded from these operators.
  //
  // System-wide uniqueness requires (node_id, device_id): two different
  // nodes can in principle report the same device_id. Use Participant
  // (which carries node_id + device) whenever crossing node boundaries.
  //
  // The constructor guarantees that device_id field is always populated.
  [[nodiscard]] bool operator==(const Device& other) const {
    return device_id == other.device_id;
  }

  [[nodiscard]] bool operator!=(const Device& other) const {
    return !(*this == other);
  }

  [[nodiscard]] bool operator<(const Device& other) const {
    return device_id < other.device_id;
  }

  /**
   * @brief Returns the local torch device index (e.g. 0 for cuda:0).
   *
   * Always reflects this process's view, even if the Device was
   * constructed via a cross-process DeviceId.
   */
  [[nodiscard]] std::int16_t LocalDeviceIndex() const {
    return static_cast<std::int16_t>(torch_device.index());
  }

  /**
   * @brief Canonical id of the physical device. Empty for non-CUDA devices.
   */
  [[nodiscard]] const DeviceId& GetDeviceId() const { return device_id; }

  torch::Device torch_device{
      torch::kCUDA};   ///< PyTorch device (always process-local)
  DeviceId device_id;  ///< Canonical id; empty for non-CUDA devices
};
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
// Hash function for Device to enable use in unordered containers
//==============================================================================
namespace std {
template <>
struct hash<setu::commons::datatypes::Device> {
  std::size_t operator()(
      const setu::commons::datatypes::Device& device) const noexcept {
    std::size_t h1 = std::hash<std::int8_t>{}(
        static_cast<std::int8_t>(device.torch_device.type()));
    std::size_t h2 = std::hash<std::int16_t>{}(
        static_cast<std::int16_t>(device.torch_device.index()));

    return h1 ^ (h2 << 1);
  }
};
}  // namespace std
//==============================================================================
