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
#include "commons/datatypes/TensorShardMetadata.h"
#include "messaging/BaseResponse.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::RequestId;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

/// @brief Response from Coordinator to NodeAgent for tensor shard registration.
/// Contains TensorShardMetadata which the NodeAgent uses to reconstruct
/// a TensorShardRef before forwarding to the client.
struct RegisterTensorShardCoordinatorResponse : public BaseResponse {
  RegisterTensorShardCoordinatorResponse(
      RequestId request_id_param,
      ErrorCode error_code_param = ErrorCode::kSuccess,
      std::optional<TensorShardMetadata> shard_metadata_param = std::nullopt)
      : BaseResponse(request_id_param, error_code_param),
        shard_metadata(std::move(shard_metadata_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "RegisterTensorShardCoordinatorResponse(request_id={}, error_code={})",
        request_id, error_code);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static RegisterTensorShardCoordinatorResponse Deserialize(
      const BinaryRange& range);

  const std::optional<TensorShardMetadata> shard_metadata;
};
using RegisterTensorShardCoordinatorResponsePtr =
    std::shared_ptr<RegisterTensorShardCoordinatorResponse>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
