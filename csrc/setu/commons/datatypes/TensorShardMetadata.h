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
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================
/**
 * @brief Metadata describing a tensor shard
 *
 * TensorShardMetadata contains the immutable metadata about a shard:
 * - A unique identifier (UUID)
 * - The specification describing the shard's shape, dtype, device, etc.
 * - The owner node ID assigned by the system
 *
 * This is used for planning, coordination, and serialization. The physical
 * device pointer and locking are handled separately in TensorShard.
 */
struct TensorShardMetadata {
  /**
   * @brief Constructs tensor shard metadata with auto-generated ID
   *
   * @param spec_param The specification for this shard
   * @param owner_param The NodeId of the owner node
   */
  TensorShardMetadata(TensorShardSpec spec_param, NodeId owner_param)
      : id(GenerateUUID()),
        spec(std::move(spec_param)),
        owner(std::move(owner_param)) {}

  /**
   * @brief Constructs tensor shard metadata with explicit ID
   *
   * @param id_param UUID identifier for this shard
   * @param spec_param The specification for this shard
   * @param owner_param The NodeId of the owner node
   *
   * @throws std::invalid_argument if shard ID is nil
   */
  TensorShardMetadata(ShardId id_param, TensorShardSpec spec_param,
                      NodeId owner_param)
      : id(id_param),
        spec(std::move(spec_param)),
        owner(std::move(owner_param)) {
    ASSERT_VALID_ARGUMENTS(!id_param.is_nil(), "Shard ID cannot be nil UUID");
  }

  /**
   * @brief Returns a string representation of the tensor shard metadata
   *
   * @return String containing shard ID, spec, and owner
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorShardMetadata(id={}, spec={}, owner={})", id,
                       spec.ToString(), owner);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static TensorShardMetadata Deserialize(const BinaryRange& range);

  const ShardId id;            ///< Unique identifier for this shard
  const TensorShardSpec spec;  ///< Specification describing the shard
  const NodeId owner;          ///< NodeId of the owner node
};
//==============================================================================
/// @brief Shared pointer to a TensorShardMetadata object
using TensorShardMetadataPtr = std::shared_ptr<TensorShardMetadata>;

/// @brief Map of shard IDs to TensorShardMetadata objects
using TensorShardMetadataMap =
    std::unordered_map<ShardId, TensorShardMetadataPtr>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
