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
#include "commons/messages/BaseResponse.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::RequestId;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

struct SubmitCopyResponse : public BaseResponse {
  SubmitCopyResponse(RequestId request_id_param,
                     CopyOperationId copy_operation_id_param,
                     ErrorCode error_code_param = ErrorCode::kSuccess)
      : BaseResponse(request_id_param, error_code_param),
        copy_operation_id(copy_operation_id_param) {}

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "SubmitCopyResponse(copy_operation_id={}, error_code={})",
        copy_operation_id, error_code);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static SubmitCopyResponse Deserialize(const BinaryRange& range);

  const CopyOperationId copy_operation_id;
};
using SubmitCopyResponsePtr = std::shared_ptr<SubmitCopyResponse>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
