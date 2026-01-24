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
#include "commons/datatypes/CopySpec.h"
#include "commons/messages/BaseRequest.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::datatypes::CopySpec;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

struct SubmitCopyRequest : public BaseRequest {
  /// @brief Constructs a request with auto-generated request ID.
  explicit SubmitCopyRequest(CopySpec copy_spec_param)
      : BaseRequest(), copy_spec(std::move(copy_spec_param)) {}

  /// @brief Constructs a request with explicit request ID (for
  /// deserialization).
  SubmitCopyRequest(RequestId request_id_param, CopySpec copy_spec_param)
      : BaseRequest(request_id_param), copy_spec(std::move(copy_spec_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format("SubmitCopyRequest(request_id={}, copy_spec={})",
                       request_id, copy_spec);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static SubmitCopyRequest Deserialize(const BinaryRange& range);

  const CopySpec copy_spec;
};
using SubmitCopyRequestPtr = std::shared_ptr<SubmitCopyRequest>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
