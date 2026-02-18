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
#include "commons/utils/Serialization.h"
//==============================================================================
#include "planner/ir/ref/RegisterRef.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
namespace setu::planner::ir::ref {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================

/// A buffer reference that can point to either a physical tensor shard
/// (ShardRef) or a physical register pool slot (RegisterRef).
///
/// LLC instructions use BufferRef as their source/destination operand type,
/// allowing uniform handling of both shard data and temporary register buffers.
struct BufferRef {
  BufferRef() = default;

  BufferRef(ShardRef shard_ref)  // NOLINT(implicit)
      : ref(std::move(shard_ref)) {}

  BufferRef(RegisterRef register_ref)  // NOLINT(implicit)
      : ref(std::move(register_ref)) {}

  [[nodiscard]] bool IsShard() const {
    return std::holds_alternative<ShardRef>(ref);
  }

  [[nodiscard]] bool IsRegister() const {
    return std::holds_alternative<RegisterRef>(ref);
  }

  [[nodiscard]] const ShardRef& AsShard() const {
    return std::get<ShardRef>(ref);
  }

  [[nodiscard]] const RegisterRef& AsRegister() const {
    return std::get<RegisterRef>(ref);
  }

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static BufferRef Deserialize(const BinaryRange& range);

  [[nodiscard]] bool operator==(const BufferRef& other) const {
    return ref == other.ref;
  }

  [[nodiscard]] bool operator!=(const BufferRef& other) const {
    return !(*this == other);
  }

  std::variant<ShardRef, RegisterRef> ref;
};

//==============================================================================
inline std::size_t hash_value(const BufferRef& buf) {
  return std::visit([](const auto& r) { return hash_value(r); }, buf.ref);
}
//==============================================================================
}  // namespace setu::planner::ir::ref
//==============================================================================
