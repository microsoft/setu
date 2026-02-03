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
#include "commons/Types.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::ir {
//==============================================================================
using setu::commons::NodeId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================
/**
 * @brief Lightweight reference to a tensor shard for use in IR instructions.
 *
 * Uniquely identifies a shard via its UUID and optionally carries debug
 * metadata (tensor name, node ID) for diagnostics and logging.
 */
struct ShardRef {
  ShardRef() = default;

  /**
   * @brief Constructs a shard reference with a shard ID and optional tensor
   * name, NodeId
   */
  ShardRef(ShardId id, std::optional<TensorName> name = std::nullopt,
           std::optional<NodeId> node = std::nullopt)
      : shard_id(std::move(id)),
        node_id(std::move(node)),
        tensor_name(std::move(name)) {}

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static ShardRef Deserialize(const BinaryRange& range);

  [[nodiscard]] bool operator==(const ShardRef& other) const {
    return shard_id == other.shard_id && tensor_name == other.tensor_name &&
           node_id == other.node_id;
  }

  ShardId shard_id;  ///< Unique UUID for the shard

  // Debug information
  std::optional<NodeId> node_id;          ///< Node where shard resides (debug)
  std::optional<TensorName> tensor_name;  ///< Parent tensor name (debug)
};
//==============================================================================
/**
 * @brief boost::hash_value specialization for ShardRef.
 *
 * Required for use with boost::concurrent_flat_map which uses boost::hash
 * by default. Defined in the same namespace for ADL lookup.
 */
inline std::size_t hash_value(const ShardRef& ref) {
  std::size_t seed = 0;
  boost::hash_combine(seed, ref.shard_id);
  if (ref.tensor_name.has_value()) {
    boost::hash_combine(seed, ref.tensor_name.value());
  }
  return seed;
}
//==============================================================================
}  // namespace setu::ir
