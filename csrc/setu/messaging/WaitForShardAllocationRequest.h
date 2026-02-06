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
#include "messaging/BaseRequest.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

struct WaitForShardAllocationRequest : public BaseRequest {
  /// @brief Constructs a request with auto-generated request ID.
  explicit WaitForShardAllocationRequest(ShardId shard_id_param)
      : BaseRequest(), shard_id(shard_id_param) {
    ASSERT_VALID_ARGUMENTS(!shard_id.is_nil(), "Shard ID cannot be nil");
  }

  /// @brief Constructs a request with explicit request ID (for
  /// deserialization).
  WaitForShardAllocationRequest(RequestId request_id_param,
                                ShardId shard_id_param)
      : BaseRequest(request_id_param), shard_id(shard_id_param) {
    ASSERT_VALID_ARGUMENTS(!shard_id.is_nil(), "Shard ID cannot be nil");
  }

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "WaitForShardAllocationRequest(request_id={}, shard_id={})", request_id,
        shard_id);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static WaitForShardAllocationRequest Deserialize(const BinaryRange& range);

  const ShardId shard_id;
};
using WaitForShardAllocationRequestPtr =
    std::shared_ptr<WaitForShardAllocationRequest>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
