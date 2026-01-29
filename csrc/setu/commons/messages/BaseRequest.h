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
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::RequestId;
//==============================================================================

/// @brief Base struct for all request messages.
/// All request types should inherit from this to ensure consistent
/// request tracking across the messaging system.
struct BaseRequest {
  /// @brief Constructs a request with auto-generated request ID.
  BaseRequest() : request_id(GenerateUUID()) {}

  /// @brief Constructs a request with explicit request ID (for
  /// deserialization).
  /// @param request_id_param The request ID to use
  explicit BaseRequest(RequestId request_id_param)
      : request_id(request_id_param) {}

  /// @brief Unique identifier for this request.
  const RequestId request_id;

 protected:
  // Protected destructor to prevent slicing when used polymorphically
  ~BaseRequest() = default;
};

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
