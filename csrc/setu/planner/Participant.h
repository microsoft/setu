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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
//==============================================================================
#include "commons/datatypes/Device.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::planner {
//==============================================================================

using setu::commons::NodeId;
using setu::commons::datatypes::Device;

//==============================================================================
/**
 * @brief Represents a participant in a distributed operation
 *
 * A Participant combines a NodeId (which node) with a Device (which device
 * on that node). This is used in the planner to identify specific devices
 * across the distributed system.
 */
struct Participant {
  Participant() = default;

  Participant(NodeId node_id_param, Device device_param)
      : node_id(std::move(node_id_param)), device(device_param) {}

  [[nodiscard]] std::string ToString() const {
    return std::format("Participant(node_id={}, device={})", node_id,
                       device.ToString());
  }

  [[nodiscard]] bool operator==(const Participant& other) const {
    return node_id == other.node_id && device == other.device;
  }

  [[nodiscard]] bool operator!=(const Participant& other) const {
    return !(*this == other);
  }

  [[nodiscard]] bool operator<(const Participant& other) const {
    return std::tie(node_id, device) < std::tie(other.node_id, other.device);
  }

  /**
   * @brief Returns the local device index
   *
   * @return Local device index from the underlying device
   */
  [[nodiscard]] std::int16_t LocalDeviceIndex() const {
    return device.LocalDeviceIndex();
  }

  void Serialize(setu::commons::BinaryBuffer& buffer) const {
    setu::commons::utils::BinaryWriter writer(buffer);
    writer.WriteFields(node_id, device);
  }

  static Participant Deserialize(const setu::commons::BinaryRange& range) {
    setu::commons::utils::BinaryReader reader(range);
    auto [node_id_val, device_val] = reader.ReadFields<NodeId, Device>();
    return Participant(node_id_val, device_val);
  }

  NodeId node_id;
  Device device;
};

using Participants = std::set<Participant>;

//==============================================================================
}  // namespace setu::planner
//==============================================================================
// Hash function for Participant to enable use in unordered containers
//==============================================================================
namespace std {
template <>
struct hash<setu::planner::Participant> {
  std::size_t operator()(const setu::planner::Participant& p) const noexcept {
    std::size_t h1 = boost::hash<setu::commons::NodeId>{}(p.node_id);
    std::size_t h2 = std::hash<setu::commons::datatypes::Device>{}(p.device);
    return h1 ^ (h2 << 1);
  }
};
}  // namespace std
//==============================================================================
