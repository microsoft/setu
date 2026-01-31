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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================
/**
 * @brief Uniquely identifies a specific shard of a tensor across the system.
 *
 * Combines the logical name of the parent tensor with a unique ShardId (UUID)
 * to facilitate lookup in distributed registries and memory managers.
 */
struct TensorShardIdentifier {
  /**
   * @brief Default constructor
   */
  TensorShardIdentifier() = default;

  /**
   * @brief Constructs an identifier for a tensor shard
   *
   * @param name The human-readable or system-assigned name of the tensor
   * @param id The unique identifier (UUID) for this specific shard
   */
  TensorShardIdentifier(TensorName name, ShardId id)
      : tensor_name(std::move(name)), shard_id(std::move(id)) {}

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static TensorShardIdentifier Deserialize(const BinaryRange& range);

  /**
   * @brief Equality comparison operator
   */
  [[nodiscard]] bool operator==(const TensorShardIdentifier& other) const {
    return tensor_name == other.tensor_name && shard_id == other.shard_id;
  }

  TensorName tensor_name;  ///< Logical name of the parent tensor
  ShardId shard_id;        ///< Unique UUID for the shard
};
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
template <>
struct std::hash<setu::commons::datatypes::TensorShardIdentifier> {
  std::size_t operator()(const setu::commons::datatypes::TensorShardIdentifier&
                             id) const noexcept {
    std::size_t h1 = std::hash<std::string>{}(id.tensor_name);
    std::size_t h2 = boost::hash<boost::uuids::uuid>{}(id.shard_id);
    return h1 ^ (h2 << 1);
  }
};
//==============================================================================
