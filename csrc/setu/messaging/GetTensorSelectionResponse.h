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
#include "commons/datatypes/TensorSelection.h"
#include "commons/utils/Serialization.h"
#include "messaging/BaseResponse.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::RequestId;
using setu::commons::datatypes::TensorSelection;
using setu::commons::enums::ErrorCode;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

struct GetTensorSelectionResponse : public BaseResponse {
  GetTensorSelectionResponse(
      RequestId request_id_param,
      ErrorCode error_code_param = ErrorCode::kSuccess,
      std::optional<TensorSelection> selection_param = std::nullopt)
      : BaseResponse(request_id_param, error_code_param),
        selection(std::move(selection_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "GetTensorSelectionResponse(request_id={}, error_code={}, "
        "has_selection={})",
        request_id, error_code, selection.has_value());
  }

  void Serialize(BinaryBuffer& buffer) const;

  static GetTensorSelectionResponse Deserialize(const BinaryRange& range);

  const std::optional<TensorSelection> selection;
};
using GetTensorSelectionResponsePtr =
    std::shared_ptr<GetTensorSelectionResponse>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================