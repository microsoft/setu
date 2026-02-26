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
#include "commons/utils/Serialization.h"
#include "messaging/BaseRequest.h"
#include "planner/Participant.h"
#include "planner/RegisterSet.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::planner::Participant;
using setu::planner::RegisterSet;
//==============================================================================

/// Sent by NodeAgent to Coordinator at startup to register its per-device
/// register sets. The Coordinator merges these into the planner backend so
/// that CIR → LLC lowering can allocate temporary registers on any device.
struct OnboardNodeAgentRequest : public BaseRequest {
  explicit OnboardNodeAgentRequest(
      std::unordered_map<Participant, RegisterSet> register_sets_param)
      : BaseRequest(),
        register_sets(std::move(register_sets_param)) {}

  OnboardNodeAgentRequest(
      RequestId request_id_param,
      std::unordered_map<Participant, RegisterSet> register_sets_param)
      : BaseRequest(request_id_param),
        register_sets(std::move(register_sets_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "OnboardNodeAgentRequest(request_id={}, num_devices={})",
        request_id, register_sets.size());
  }

  void Serialize(BinaryBuffer& buffer) const;
  static OnboardNodeAgentRequest Deserialize(const BinaryRange& range);

  const std::unordered_map<Participant, RegisterSet> register_sets;
};

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
