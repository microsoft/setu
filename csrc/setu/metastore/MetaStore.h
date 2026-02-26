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
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/datatypes/TensorSpec.h"
#include "metastore/datatypes/TensorMetadata.h"
//==============================================================================
namespace setu::metastore {
//==============================================================================
using setu::commons::NodeId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorShardMetadataMap;
using setu::commons::datatypes::TensorShardMetadataPtr;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::datatypes::TensorShardSpecPtr;
using setu::commons::datatypes::TensorSpec;
using setu::commons::datatypes::TensorSpecMap;
using setu::metastore::datatypes::TensorMetadata;
using setu::metastore::datatypes::TensorMetadataMap;
using setu::metastore::datatypes::TensorMetadataPtr;
//==============================================================================
/**
 * @brief Metadata store for managing tensor shard registrations.
 *
 * MetaStore tracks all registered tensor shards in the system. It supports
 * incremental registration (shards arrive one at a time) and builds cached
 * TensorMetadata once all shards for a tensor are present.
 *
 * ## Deregistration (two-phase)
 *
 * Shard removal is split into two phases because in-flight copy operations
 * may still reference the shard metadata:
 *
 *   1. Mark. MarkTensorDeregistered(name) sets a flag that causes
 *      IsTensorDeregistered to return true. The Coordinator uses this to
 *      immediately reject new registrations, copies, and pulls for the
 *      tensor. No metadata is removed yet.
 *
 *   2. Remove. DeregisterShards(...) actually deletes shard metadata,
 *      updates sizes, and cleans up caches. The Coordinator calls this
 *      only after all in-flight copies touching the tensor have finished.
 *
 * This ensures in-flight operations can still read shard metadata while
 * no new operations are accepted on a tensor being torn down.
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
   * @return TensorShardMetadataPtr containing the assigned shard ID and
   * metadata
   */
  [[nodiscard]] TensorShardMetadataPtr RegisterTensorShard(
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
   * Returns nullptr if the tensor is not found or not fully registered.
   *
   * @param tensor_name The name of the tensor to query
   * @return Shared pointer to const TensorMetadata if fully registered, nullptr
   * otherwise
   */
  [[nodiscard]] TensorMetadataPtr GetTensorMetadata(
      const TensorName& tensor_name /*[in]*/);

  /**
   * @brief Returns the TensorSpec for a tensor
   *
   * TensorSpec is cached on first shard registration, so this is available
   * before all shards are registered. Returns nullptr if no shard has been
   * registered for this tensor.
   *
   * @param tensor_name The name of the tensor to query
   * @return Pointer to the TensorSpec if any shard has been registered, nullptr
   * otherwise
   */
  [[nodiscard]] const TensorSpec* GetTensorSpec(
      const TensorName& tensor_name /*[in]*/) const;

  /**
   * @brief Checks if any shards of a tensor have been deregistered
   *
   * Returns true if DeregisterShards has been called for this tensor,
   * indicating the tensor is partially freed.
   *
   * @param tensor_name The name of the tensor to check
   * @return true if any shards have been deregistered, false otherwise
   */
  [[nodiscard]] bool IsTensorDeregistered(
      const TensorName& tensor_name /*[in]*/) const;

  /**
   * @brief Marks a tensor as deregistered without removing its shards
   *
   * Sets the has_deregistered_shards flag so that IsTensorDeregistered returns
   * true and new registrations/operations are rejected. The actual shard
   * removal happens later via DeregisterShards. This is used by the
   * Coordinator to immediately block new operations when a deregistration
   * request arrives, even if the actual removal is deferred.
   *
   * @param tensor_name The name of the tensor to mark as deregistered
   */
  void MarkTensorDeregistered(const TensorName& tensor_name /*[in]*/);

  /**
   * @brief Deregisters tensor shards from the metadata store
   *
   * Removes the specified shards from each tensor's registration data,
   * updates registered sizes, and invalidates caches. If all shards for
   * a tensor are removed, the entire tensor entry is cleaned up.
   *
   * @param shards_by_tensor Map of tensor name to shard IDs to deregister
   */
  void DeregisterShards(
      const std::unordered_map<TensorName, std::vector<ShardId>>&
          shards_by_tensor /*[in]*/);

 private:
  /// Registered shard data: expected size, registered size, and shard metadata
  struct RegisteredShardsData {
    std::size_t expected_size{0};
    std::size_t registered_size{0};
    TensorShardMetadataMap shards;
    bool has_deregistered_shards{false};
  };

  /**
   * @brief Validates a shard registration against existing shards
   *
   * Checks:
   * 1. Dtype matches existing shards
   * 2. Dimension count, names, and sizes match
   * 3. No overlap with any existing shard
   *
   * @param shard_spec The shard specification to validate
   * @param registered_data Existing registered data to validate against
   * @return true if validation passes, false otherwise
   */
  [[nodiscard]] bool ValidateShardRegistration(
      const TensorShardSpec& shard_spec /*[in]*/,
      const RegisteredShardsData& registered_data /*[in]*/) const;

  mutable std::recursive_mutex mutex_;
  std::unordered_map<TensorName, RegisteredShardsData> registered_shards_data_;
  TensorMetadataMap tensor_metadata_cache_;
  TensorSpecMap tensor_spec_cache_;
};
//==============================================================================
}  // namespace setu::metastore
//==============================================================================
