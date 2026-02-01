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
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/Device.h"
#include "ir/Instruction.h"
#include "metastore/MetaStore.h"
//==============================================================================
namespace setu::planner {
//==============================================================================

using setu::commons::NodeId;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::Device;
using setu::ir::Program;
using setu::metastore::MetaStore;

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

  /**
   * @brief Returns the local device index
   *
   * @return Local device index from the underlying device
   */
  [[nodiscard]] std::int16_t LocalDeviceIndex() const {
    return device.LocalDeviceIndex();
  }

  NodeId node_id;
  Device device;
};

using Participants = std::vector<Participant>;
//==============================================================================
}  // namespace setu::planner
//==============================================================================
// Hash function for Participant to enable use in unordered containers
//==============================================================================
namespace std {
template <>
struct hash<setu::planner::Participant> {
  std::size_t operator()(
      const setu::planner::Participant& participant) const noexcept {
    std::size_t h1 = boost::hash<boost::uuids::uuid>{}(participant.node_id);
    std::size_t h2 =
        std::hash<setu::commons::datatypes::Device>{}(participant.device);

    return h1 ^ (h2 << 1);
  }
};
}  // namespace std
//==============================================================================
namespace setu::planner {
//==============================================================================

struct Plan {
  std::unordered_map<NodeId, Plan> Fragments();

  [[nodiscard]] std::string ToString() const {
    return std::format("Plan(participants={}, programs={})",
                       participants.size(), program.size());
  }

  Participants participants;
  std::unordered_map<Participant, Program> program;
};

class Planner {
 public:
  virtual ~Planner() = default;
  virtual Plan Compile(CopySpec& spec, MetaStore& metastore) = 0;
};
//==============================================================================
}  // namespace setu::planner
//==============================================================================
