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
#include "messaging/DeregisterShardsRequest.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

void DeregisterShardsRequest::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(request_id, shards_by_tensor);
}

DeregisterShardsRequest DeregisterShardsRequest::Deserialize(
    const BinaryRange& range) {
  BinaryReader reader(range);
  auto [request_id_val, shards_by_tensor_val] =
      reader.ReadFields<RequestId,
                        std::unordered_map<TensorName, std::vector<ShardId>>>();
  return DeregisterShardsRequest(request_id_val,
                                 std::move(shards_by_tensor_val));
}

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
