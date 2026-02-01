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
#include "metastore/MetaStore.h"
//==============================================================================
#include "commons/BoostCommon.h"
#include "commons/datatypes/TensorDim.h"
//==============================================================================
namespace setu::metastore {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::datatypes::TensorDim;
using setu::commons::datatypes::TensorDimMap;
//==============================================================================
TensorShardRef MetaStore::RegisterTensorShard(const TensorShardSpec& shard_spec,
                                              const NodeId& owner_node_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  ShardId shard_id = GenerateUUID();

  auto& tensor_data = tensor_shards_data_[shard_spec.name];

  // Store the shard spec and owner
  tensor_data.shards_specs.emplace(
      shard_id, std::make_shared<TensorShardSpec>(shard_spec));
  tensor_data.shard_owners.emplace(shard_id, owner_node_id);

  // Convert TensorDimSpec vector to TensorDimMap for the reference
  TensorDimMap dims;
  for (const auto& dim_spec : shard_spec.dims) {
    dims.emplace(dim_spec.name, TensorDim(dim_spec.name, dim_spec.size));
  }

  TensorShardRef shard_ref(shard_spec.name, shard_id, dims);

  // Calculate and track sizes
  std::size_t shard_num_elements = shard_spec.GetNumElements();

  // Initialize expected size on first shard registration
  if (tensor_data.expected_size == 0) {
    std::size_t total_tensor_size = 1;
    for (const auto& dim_spec : shard_spec.dims) {
      total_tensor_size *= dim_spec.size;
    }
    tensor_data.expected_size = total_tensor_size;
  }

  tensor_data.registered_size += shard_num_elements;

  LOG_DEBUG(
      "Registered tensor shard: id={}, name={}, num_dims={}, "
      "shard_elements={}, registered={}/{}",
      shard_id, shard_spec.name, dims.size(), shard_num_elements,
      tensor_data.registered_size, tensor_data.expected_size);

  return shard_ref;
}
//==============================================================================
bool MetaStore::AllShardsRegistered(const TensorName& tensor_name) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = tensor_shards_data_.find(tensor_name);
  if (it == tensor_shards_data_.end()) {
    return false;
  }
  return it->second.registered_size == it->second.expected_size;
}
//==============================================================================
std::size_t MetaStore::GetNumShardsForTensor(
    const TensorName& tensor_name) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = tensor_shards_data_.find(tensor_name);
  if (it != tensor_shards_data_.end()) {
    return it->second.shards_specs.size();
  }
  return 0;
}
//==============================================================================
TensorMetadataPtr MetaStore::GetTensorMetadata(
    const TensorName& tensor_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Check cache first
  auto cache_it = tensor_metadata_cache_.find(tensor_name);
  if (cache_it != tensor_metadata_cache_.end()) {
    return cache_it->second;
  }

  // Check if all shards are registered
  if (!AllShardsRegistered(tensor_name)) {
    return nullptr;
  }

  auto tensor_it = tensor_shards_data_.find(tensor_name);
  ASSERT_VALID_RUNTIME(tensor_it != tensor_shards_data_.end(),
                       "Tensor {} should exist if all shards are registered",
                       tensor_name);

  const auto& tensor_data = tensor_it->second;
  const auto& shards = tensor_data.shards_specs;
  const auto& shard_owners = tensor_data.shard_owners;
  ASSERT_VALID_RUNTIME(!shards.empty(),
                       "Tensor {} should have at least one shard", tensor_name);

  // Get dims and dtype from first shard
  const auto& first_shard_spec = shards.begin()->second;
  TensorDimMap dims;
  for (const auto& dim_spec : first_shard_spec->dims) {
    dims.emplace(dim_spec.name, TensorDim(dim_spec.name, dim_spec.size));
  }

  // Build and cache TensorMetadata
  auto metadata = std::make_shared<TensorMetadata>(
      tensor_name, dims, first_shard_spec->dtype, shards, shard_owners);
  tensor_metadata_cache_.emplace(tensor_name, metadata);

  return metadata;
}
//==============================================================================
}  // namespace setu::metastore
//==============================================================================