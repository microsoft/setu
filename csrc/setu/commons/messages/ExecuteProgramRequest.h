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
#include "commons/messages/BaseRequest.h"
#include "commons/utils/Serialization.h"
#include "coordinator/datatypes/Program.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::coordinator::datatypes::Program;
//==============================================================================

struct ExecuteProgramRequest : public BaseRequest {
  /// @brief Constructs a request with auto-generated request ID.
  explicit ExecuteProgramRequest(Program program_param)
      : BaseRequest(), program(std::move(program_param)) {}

  /// @brief Constructs a request with explicit request ID (for
  /// deserialization).
  ExecuteProgramRequest(RequestId request_id_param, Program program_param)
      : BaseRequest(request_id_param), program(std::move(program_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format("ExecuteProgramRequest(request_id={}, program={})",
                       request_id, program.ToString());
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(request_id, program);
  }

  static ExecuteProgramRequest Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto [request_id_val, program_val] =
        reader.ReadFields<RequestId, Program>();
    return ExecuteProgramRequest(request_id_val, std::move(program_val));
  }

  const Program program;
};
using ExecuteProgramRequestPtr = std::shared_ptr<ExecuteProgramRequest>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
