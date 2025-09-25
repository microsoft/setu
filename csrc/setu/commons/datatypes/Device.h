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
#include "commons/Types.h"
#include "commons/enums/Enums.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::enums::DeviceKind;
//==============================================================================
/**
 * @brief Represents a physical compute device in the distributed system
 *
 * Device encapsulates information about a physical compute device including
 * its type (CPU, GPU, etc.) and its position within the distributed system
 * hierarchy (node rank, global device rank, and local device rank).
 */
struct Device {
  /**
   * @brief Default constructor for empty device
   */
  Device() = default;

  /**
   * @brief Constructs a device with all identifying information
   *
   * @param kind_param Type of device (CPU, GPU, etc.)
   * @param node_rank_param Rank of the node containing this device
   * @param device_rank_param Global rank of this device across all nodes
   * @param local_device_rank_param Local rank of this device within its node
   */
  Device(DeviceKind kind_param, NodeRank node_rank_param,
         DeviceRank device_rank_param, LocalDeviceRank local_device_rank_param)
      : kind(kind_param),
        node_rank(node_rank_param),
        device_rank(device_rank_param),
        local_device_rank(local_device_rank_param) {}

  /**
   * @brief Returns a string representation of the device
   *
   * @return String containing device kind and all rank information
   */
  [[nodiscard]] std::string ToString() const {
    return std::format(
        "Device(kind={}, node_rank={}, device_rank={}, local_device_rank={})",
        kind, node_rank, device_rank, local_device_rank);
  }

  const DeviceKind kind;         ///< Type of device (CPU, GPU, etc.)
  const NodeRank node_rank;      ///< Rank of the node containing this device
  const DeviceRank device_rank;  ///< Global rank across all devices
  const LocalDeviceRank local_device_rank;  ///< Local rank within the node
};
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
