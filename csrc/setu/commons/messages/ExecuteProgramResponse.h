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
#include "setu/commons/StdCommon.h"
//==============================================================================
#include "setu/commons/Types.h"
#include "setu/commons/messages/BaseResponse.h"
#include "setu/commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::RequestId;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

struct ExecuteProgramResponse : public BaseResponse {
  explicit ExecuteProgramResponse(
      RequestId request_id_param,
      ErrorCode error_code_param = ErrorCode::kSuccess)
      : BaseResponse(request_id_param, error_code_param) {}

  [[nodiscard]] std::string ToString() const {
    return std::format("ExecuteProgramResponse(error_code={})", error_code);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static ExecuteProgramResponse Deserialize(const BinaryRange& range);
};
using ExecuteProgramResponsePtr = std::shared_ptr<ExecuteProgramResponse>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
