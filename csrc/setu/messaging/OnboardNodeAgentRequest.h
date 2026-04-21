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
#include "commons/utils/Serialization.h"
#include "messaging/BaseRequest.h"
#include "planner/Participant.h"
#include "planner/RegisterSet.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::datatypes::DeviceId;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::planner::Participant;
using setu::planner::RegisterSet;
//==============================================================================

/// Directional P2P-capable device pair, keyed on canonical DeviceId so the
/// coordinator can match against devices reported by any node agent without
/// needing a process-local DeviceMap.
struct P2PPair {
  DeviceId src;
  DeviceId dst;

  void Serialize(BinaryBuffer& buffer) const {
    setu::commons::utils::BinaryWriter writer(buffer);
    writer.WriteFields(src, dst);
  }

  static P2PPair Deserialize(const BinaryRange& range) {
    setu::commons::utils::BinaryReader reader(range);
    auto [s, d] = reader.ReadFields<DeviceId, DeviceId>();
    return {std::move(s), std::move(d)};
  }
};

/// Sent by NodeAgent to Coordinator at startup to register its per-device
/// register sets and P2P topology. The Coordinator merges these into the
/// planner so that CIR → LLC lowering can allocate temporary registers and
/// decide between Pull vs NCCL Send/Receive.
struct OnboardNodeAgentRequest : public BaseRequest {
  using P2PPairs = std::vector<P2PPair>;

  explicit OnboardNodeAgentRequest(
      std::unordered_map<Participant, RegisterSet> register_sets_param,
      P2PPairs p2p_pairs_param = {})
      : BaseRequest(),
        register_sets(std::move(register_sets_param)),
        p2p_pairs(std::move(p2p_pairs_param)) {}

  OnboardNodeAgentRequest(
      RequestId request_id_param,
      std::unordered_map<Participant, RegisterSet> register_sets_param,
      P2PPairs p2p_pairs_param = {})
      : BaseRequest(request_id_param),
        register_sets(std::move(register_sets_param)),
        p2p_pairs(std::move(p2p_pairs_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "OnboardNodeAgentRequest(request_id={}, num_devices={}, "
        "num_p2p_pairs={})",
        request_id, register_sets.size(), p2p_pairs.size());
  }

  void Serialize(BinaryBuffer& buffer) const;
  static OnboardNodeAgentRequest Deserialize(const BinaryRange& range);

  const std::unordered_map<Participant, RegisterSet> register_sets;

  /// Directional P2P-capable device pairs (src_local_idx, dst_local_idx)
  /// reported by the NodeAgent after probing cudaDeviceCanAccessPeer.
  const P2PPairs p2p_pairs;
};

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
