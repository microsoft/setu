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
#include <nccl.h>
//==============================================================================
#include "setu/commons/StdCommon.h"
#include "setu/commons/Types.h"
#include "setu/commons/utils/Serialization.h"
//==============================================================================
namespace setu::ir {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

struct UseCommInstruction {
  explicit UseCommInstruction(ncclUniqueId comm_id)
      : comm_id(std::move(comm_id)) {}

  ~UseCommInstruction() = default;
  UseCommInstruction(const UseCommInstruction&) = default;
  UseCommInstruction& operator=(const UseCommInstruction&) = default;
  UseCommInstruction(UseCommInstruction&&) = default;
  UseCommInstruction& operator=(UseCommInstruction&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static UseCommInstruction Deserialize(const BinaryRange& range);

  ncclUniqueId comm_id;
};

//==============================================================================
}  // namespace setu::ir
//==============================================================================
