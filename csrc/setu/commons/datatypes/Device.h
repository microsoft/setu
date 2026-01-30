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
 * @brief Represents a physical compute device in the distributed system
 *
 * Device encapsulates information about a physical compute device including
 * its type (CPU, GPU, etc.) via torch::Device, and its position within the
 * distributed system hierarchy (node rank and global device rank).
 */
struct Device {
  /**
   * @brief Default constructor for empty device
   */
  Device() = default;

  /**
   * @brief Constructs a device with all identifying information
   *
   * @param node_id_param ID of the node containing this device
   * @param device_rank_param Global rank of this device across all nodes
   * @param torch_device_param PyTorch device (type + local index, e.g., cuda:0)
   */
  Device(NodeId node_id_param, DeviceRank device_rank_param,
         torch::Device torch_device_param)
      : node_id(node_id_param),
        device_rank(device_rank_param),
        torch_device(torch_device_param) {}

  /**
   * @brief Returns a string representation of the device
   *
   * @return String containing node ID, device rank, and torch device info
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("Device(node_id={}, device_rank={}, torch_device={})",
                       node_id, device_rank, torch_device.str());
  }

  void Serialize(BinaryBuffer& buffer) const;

  static Device Deserialize(const BinaryRange& range);

  /**
   * @brief Equality comparison operator
   *
   * @param other Device to compare with
   * @return true if devices are equal, false otherwise
   */
  [[nodiscard]] bool operator==(const Device& other) const {
    return node_id == other.node_id && device_rank == other.device_rank &&
           torch_device == other.torch_device;
  }

  /**
   * @brief Inequality comparison operator
   *
   * @param other Device to compare with
   * @return true if devices are not equal, false otherwise
   */
  [[nodiscard]] bool operator!=(const Device& other) const {
    return !(*this == other);
  }

  const NodeId node_id;              ///< ID of the node containing this device
  const DeviceRank device_rank;      ///< Global rank across all devices
  const torch::Device torch_device;  ///< PyTorch device (type + local index)
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
    std::size_t h1 = boost::hash<boost::uuids::uuid>{}(device.node_id);
    std::size_t h2 = std::hash<std::size_t>{}(device.device_rank);
    std::size_t h3 = std::hash<std::int8_t>{}(
        static_cast<std::int8_t>(device.torch_device.type()));
    std::size_t h4 = std::hash<std::int16_t>{}(
        static_cast<std::int16_t>(device.torch_device.index()));

    // Combine hashes using XOR and bit shifting
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
  }
};
}  // namespace std
//==============================================================================
