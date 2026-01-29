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
#include "commons/messages/GetTensorHandleRequest.h"
#include "commons/messages/GetTensorHandleResponse.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

void GetTensorHandleRequest::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(request_id, tensor_name);
}

GetTensorHandleRequest GetTensorHandleRequest::Deserialize(
    const BinaryRange& range) {
  BinaryReader reader(range);
  auto [request_id_val, tensor_name_val] =
      reader.ReadFields<RequestId, TensorName>();
  return GetTensorHandleRequest(request_id_val, tensor_name_val);
}

void GetTensorHandleResponse::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(request_id, error_code, tensor_ipc_spec);
}

GetTensorHandleResponse GetTensorHandleResponse::Deserialize(
    const BinaryRange& range) {
  BinaryReader reader(range);
  auto [request_id_val, error_code_val, tensor_ipc_spec_val] =
      reader.ReadFields<RequestId, ErrorCode, std::optional<TensorIPCSpec>>();
  return GetTensorHandleResponse(request_id_val, error_code_val,
                                 std::move(tensor_ipc_spec_val));
}

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
