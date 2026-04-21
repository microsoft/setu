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
#include "commons/Types.h"
#include "commons/datatypes/DeviceId.h"
//==============================================================================
#include "planner/Participant.h"
#include "planner/RegisterSet.h"
#include "planner/hints/HintStore.h"
#include "planner/ir/cir/Value.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================

/// Represents a single directional P2P-capable device pair within a node,
/// keyed on canonical DeviceId (not process-local torch indices).
struct P2PDevicePair {
  setu::commons::datatypes::DeviceId src;
  setu::commons::datatypes::DeviceId dst;

  bool operator<(const P2PDevicePair& other) const {
    return std::tie(src, dst) < std::tie(other.src, other.dst);
  }

  bool operator==(const P2PDevicePair& other) const {
    return src == other.src && dst == other.dst;
  }
};

/// Per-node map of directional P2P-capable device pairs.
/// Key: NodeId.  Value: set of (src_device_id, dst_device_id) where
/// cudaDeviceCanAccessPeer returned true.
using P2PAccessMap =
    std::unordered_map<setu::commons::NodeId, std::set<P2PDevicePair>>;

/// Immutable context passed to every pass and the backend at run time.
///
/// Contains per-operation hints (routing, bandwidth), global compiler
/// configuration (register pool sizes), and node-level P2P topology.
/// Passes and backends read what they need and ignore the rest.
struct PassContext {
  const setu::planner::hints::HintStore& hints;
  const std::unordered_map<setu::planner::ir::cir::Device,
                           setu::planner::RegisterSet>& register_sets;
  const P2PAccessMap& p2p_access;

  /// Returns true if dst can directly pull from src via cudaMemcpyPeerAsync.
  [[nodiscard]] bool HasP2PAccess(const setu::planner::Participant& src,
                                  const setu::planner::Participant& dst) const {
    auto it = p2p_access.find(src.node_id);
    if (it == p2p_access.end()) return false;
    return it->second.contains(
        {src.device.GetDeviceId(), dst.device.GetDeviceId()});
  }
};

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
