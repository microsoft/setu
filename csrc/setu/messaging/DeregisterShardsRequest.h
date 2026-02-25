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
#include "commons/Types.h"
#include "commons/utils/Serialization.h"
#include "messaging/BaseRequest.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

/// @brief Request to deregister tensor shards on disconnect.
/// Sent from Client → NodeAgent and forwarded NodeAgent → Coordinator.
struct DeregisterShardsRequest : public BaseRequest {
  /// @brief Constructs a request with auto-generated request ID.
  explicit DeregisterShardsRequest(
      std::unordered_map<TensorName, std::vector<ShardId>>
          shards_by_tensor_param)
      : BaseRequest(), shards_by_tensor(std::move(shards_by_tensor_param)) {
    ASSERT_VALID_ARGUMENTS(!shards_by_tensor.empty(),
                           "Shards map cannot be empty");
  }

  /// @brief Constructs a request with explicit request ID (for
  /// deserialization).
  DeregisterShardsRequest(RequestId request_id_param,
                          std::unordered_map<TensorName, std::vector<ShardId>>
                              shards_by_tensor_param)
      : BaseRequest(request_id_param),
        shards_by_tensor(std::move(shards_by_tensor_param)) {
    ASSERT_VALID_ARGUMENTS(!shards_by_tensor.empty(),
                           "Shards map cannot be empty");
  }

  [[nodiscard]] std::string ToString() const {
    std::size_t total_shards = 0;
    for (const auto& [name, ids] : shards_by_tensor) {
      total_shards += ids.size();
    }
    return std::format(
        "DeregisterShardsRequest(request_id={}, num_tensors={}, "
        "total_shards={})",
        request_id, shards_by_tensor.size(), total_shards);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static DeregisterShardsRequest Deserialize(const BinaryRange& range);

  const std::unordered_map<TensorName, std::vector<ShardId>> shards_by_tensor;
};
using DeregisterShardsRequestPtr = std::shared_ptr<DeregisterShardsRequest>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
