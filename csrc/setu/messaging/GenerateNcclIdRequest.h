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
#include "commons/utils/Serialization.h"
#include "messaging/BaseRequest.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::RequestId;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

/// @brief Request sent from Coordinator to the rank-0 NodeAgent to generate
/// an ncclUniqueId. The NodeAgent calls ncclGetUniqueId() and sends back
/// a GenerateNcclIdResponse.
struct GenerateNcclIdRequest : public BaseRequest {
  /// @brief Constructs a request with auto-generated request ID.
  GenerateNcclIdRequest() : BaseRequest() {}

  /// @brief Constructs a request with explicit request ID (for
  /// deserialization).
  explicit GenerateNcclIdRequest(RequestId request_id_param)
      : BaseRequest(request_id_param) {}

  [[nodiscard]] std::string ToString() const {
    return std::format("GenerateNcclIdRequest(request_id={})", request_id);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static GenerateNcclIdRequest Deserialize(const BinaryRange& range);
};

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
