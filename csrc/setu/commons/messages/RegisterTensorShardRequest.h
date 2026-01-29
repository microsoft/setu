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
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/messages/BaseRequest.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

struct RegisterTensorShardRequest : public BaseRequest {
  /// @brief Constructs a request with auto-generated request ID.
  explicit RegisterTensorShardRequest(TensorShardSpec tensor_shard_spec_param)
      : BaseRequest(), tensor_shard_spec(std::move(tensor_shard_spec_param)) {
    ASSERT_VALID_ARGUMENTS(!tensor_shard_spec.name.empty(),
                           "Tensor name cannot be empty");
  }

  /// @brief Constructs a request with explicit request ID (for
  /// deserialization).
  RegisterTensorShardRequest(RequestId request_id_param,
                             TensorShardSpec tensor_shard_spec_param)
      : BaseRequest(request_id_param),
        tensor_shard_spec(std::move(tensor_shard_spec_param)) {
    ASSERT_VALID_ARGUMENTS(!tensor_shard_spec.name.empty(),
                           "Tensor name cannot be empty");
  }

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "RegisterTensorShardRequest(request_id={}, tensor_name={})", request_id,
        tensor_shard_spec.name);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static RegisterTensorShardRequest Deserialize(const BinaryRange& range);

  const TensorShardSpec tensor_shard_spec;
};
using RegisterTensorShardRequestPtr =
    std::shared_ptr<RegisterTensorShardRequest>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
