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
#include "commons/enums/Enums.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::RequestId;
using setu::commons::enums::ErrorCode;
//==============================================================================

/// @brief Base struct for all response messages.
/// All response types should inherit from this to ensure consistent error
/// handling across the messaging system.
struct BaseResponse {
  /// @brief Constructs a response with request ID and error code.
  /// @param request_id_param The request ID this response corresponds to
  /// @param error_code_param The error code indicating operation status
  explicit BaseResponse(RequestId request_id_param,
                        ErrorCode error_code_param = ErrorCode::kSuccess)
      : request_id(request_id_param), error_code(error_code_param) {}

  /// @brief Check if the response indicates success.
  [[nodiscard]] bool IsSuccess() const {
    return error_code == ErrorCode::kSuccess;
  }

  /// @brief Check if the response indicates an error.
  [[nodiscard]] bool IsError() const {
    return error_code != ErrorCode::kSuccess;
  }

  /// @brief The request ID this response corresponds to.
  const RequestId request_id;

  /// @brief The error code indicating the status of the operation.
  const ErrorCode error_code;

 protected:
  // Protected destructor to prevent slicing when used polymorphically
  ~BaseResponse() = default;
};

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
