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
#include "commons/Types.h"
//==============================================================================
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
//==============================================================================
namespace setu::metastore {
//==============================================================================
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
//==============================================================================
/**
 * @brief Metadata store for managing tensor shard registrations
 *
 * MetaStore is responsible for tracking all registered tensor shards in the
 * system. It assigns unique identifiers to shards and provides lookup
 * capabilities for retrieving shard metadata.
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
   * @return TensorShardRef containing the assigned shard ID and metadata
   */
  [[nodiscard]] TensorShardRef RegisterTensorShard(
      const TensorShardSpec& shard_spec /*[in]*/);

  /**
   * @brief Looks up a tensor shard by its unique identifier
   *
   * @param shard_id The UUID of the shard to look up
   * @return Optional containing the TensorShardRef if found, empty otherwise
   */
  [[nodiscard]] std::optional<TensorShardRef> GetShardById(
      const ShardId& shard_id /*[in]*/) const;

  /**
   * @brief Gets all shards registered for a given tensor name
   *
   * @param tensor_name The name of the tensor to query
   * @return Vector of TensorShardRef for all shards of the specified tensor
   */
  [[nodiscard]] std::vector<TensorShardRef> GetShardsByTensorName(
      const TensorName& tensor_name /*[in]*/) const;

  /**
   * @brief Returns the total number of registered shards
   *
   * @return Number of shards in the metadata store
   */
  [[nodiscard]] std::size_t GetNumShards() const;

  [[nodiscard]] bool AllShardsRegistered(const TensorName& tensor_name) const;

  /**
   * @brief Returns the number of shards registered for a given tensor
   *
   * @param tensor_name The name of the tensor to query
   * @return Number of shards registered for this tensor
   */
  [[nodiscard]] std::size_t GetNumShardsForTensor(
      const TensorName& tensor_name /*[in]*/) const;

 private:
  /// Map from shard ID to shard reference
  std::unordered_map<ShardId, TensorShardRef> shards_by_id_;

  /// Map from tensor name to list of shard IDs for that tensor
  std::unordered_map<TensorName, std::vector<ShardId>> shards_by_tensor_name_;

  /// Map from tensor name to expected total number of elements
  std::unordered_map<TensorName, std::size_t> tensor_expected_size_;

  /// Map from tensor name to currently registered number of elements
  std::unordered_map<TensorName, std::size_t> tensor_registered_size_;
};
//==============================================================================
}  // namespace setu::metastore
//==============================================================================