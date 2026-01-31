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
#include "commons/ClassTraits.h"
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/Types.h"
//==============================================================================
#include "commons/datatypes/TensorDim.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "metastore/datatypes/TensorMetadata.h"
//==============================================================================
namespace setu::metastore {
//==============================================================================
using setu::commons::NodeId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::datatypes::TensorShardSpecPtr;
using setu::metastore::datatypes::TensorMetadata;
//==============================================================================
/**
 * @brief Metadata store for managing tensor shard registrations
 *
 * MetaStore is responsible for tracking all registered tensor shards in the
 * system.
 */
class MetaStore {
 public:
  MetaStore() = default;

  /**
   * @brief Registers a new tensor shard in the metadata store
   *
   * Creates a new shard registration from the provided specification. A unique
   * shard ID is generated and assigned to the shard. The shard metadata is
   * stored for future lookups.
   *
   * @param shard_spec The specification describing the tensor shard to register
   * @param owner_node_id The NodeId of the NodeAgent that owns this shard
   * @return TensorShardRef containing the assigned shard ID and metadata
   */
  [[nodiscard]] TensorShardRef RegisterTensorShard(
      const TensorShardSpec& shard_spec /*[in]*/,
      const NodeId& owner_node_id /*[in]*/);

  [[nodiscard]] bool AllShardsRegistered(const TensorName& tensor_name) const;

  /**
   * @brief Returns the number of shards registered for a given tensor
   *
   * @param tensor_name The name of the tensor to query
   * @return Number of shards registered for this tensor
   */
  [[nodiscard]] std::size_t GetNumShardsForTensor(
      const TensorName& tensor_name /*[in]*/) const;

  /**
   * @brief Returns the tensor metadata for a fully registered tensor
   *
   * Builds and caches TensorMetadata when all shards have been registered.
   * Returns std::nullopt if the tensor is not found or not fully registered.
   *
   * @param tensor_name The name of the tensor to query
   * @return Optional containing TensorMetadata if fully registered, nullopt
   * otherwise
   */
  [[nodiscard]] std::optional<TensorMetadata> GetTensorMetadata(
      const TensorName& tensor_name /*[in]*/);

 private:
  /// Tensor shard data: expected size, registered size, shards, and owners
  struct TensorShardsData {
    std::size_t expected_size{0};
    std::size_t registered_size{0};
    std::unordered_map<ShardId, TensorShardSpecPtr> shards_specs;
    std::unordered_map<ShardId, NodeId> shard_owners;
  };

  mutable std::recursive_mutex mutex_;
  std::unordered_map<TensorName, TensorShardsData> tensor_shards_data_;
  std::unordered_map<TensorName, TensorMetadata> tensor_metadata_cache_;
};
//==============================================================================
}  // namespace setu::metastore
//==============================================================================