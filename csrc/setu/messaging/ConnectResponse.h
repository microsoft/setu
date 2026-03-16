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
#include "commons/utils/Serialization.h"
#include "messaging/BaseResponse.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::enums::ErrorCode;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

/// @brief Response from NodeAgent to Client after a ConnectRequest,
/// providing the shared-memory completion ring name and capacity.
struct ConnectResponse : public BaseResponse {
  ConnectResponse(RequestId request_id_param,
                  ErrorCode error_code_param = ErrorCode::kSuccess,
                  std::string completion_ring_shm_name_param = "",
                  std::uint32_t completion_ring_capacity_param = 0)
      : BaseResponse(request_id_param, error_code_param),
        completion_ring_shm_name(std::move(completion_ring_shm_name_param)),
        completion_ring_capacity(completion_ring_capacity_param) {}

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "ConnectResponse(request_id={}, error_code={}, shm_name={}, "
        "capacity={})",
        request_id, error_code, completion_ring_shm_name,
        completion_ring_capacity);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static ConnectResponse Deserialize(const BinaryRange& range);

  const std::string completion_ring_shm_name;
  const std::uint32_t completion_ring_capacity;
};
using ConnectResponsePtr = std::shared_ptr<ConnectResponse>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
