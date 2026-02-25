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
using setu::commons::datatypes::TensorDim;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorSpec;
//==============================================================================
TensorShardMetadataPtr MetaStore::RegisterTensorShard(
    const TensorShardSpec& shard_spec, const NodeId& owner_node_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto& registered_data = registered_shards_data_[shard_spec.name];

  // Validate shard against existing registrations
  if (!ValidateShardRegistration(shard_spec, registered_data)) {
    return nullptr;
  }

  // Create TensorShardMetadata with auto-generated ID
  auto shard_metadata =
      std::make_shared<TensorShardMetadata>(shard_spec, owner_node_id);
  ShardId shard_id = shard_metadata->id;

  // Store the shard metadata
  registered_data.shards.emplace(shard_id, shard_metadata);

  // Calculate and track sizes
  std::size_t shard_num_elements = shard_spec.GetNumElements();

  // Initialize expected size and cache TensorSpec on first shard registration
  if (registered_data.expected_size == 0) {
    std::size_t total_tensor_size = 1;
    TensorDimMap dims;
    for (const auto& dim_spec : shard_spec.dims) {
      total_tensor_size *= dim_spec.size;
      dims.emplace(dim_spec.name, TensorDim(dim_spec.name, dim_spec.size));
    }
    registered_data.expected_size = total_tensor_size;
    tensor_spec_cache_.emplace(
        shard_spec.name,
        TensorSpec(shard_spec.name, std::move(dims), shard_spec.dtype));
  }

  registered_data.registered_size += shard_num_elements;

  LOG_DEBUG(
      "Registered tensor shard: id={}, name={}, num_dims={}, "
      "shard_elements={}, registered={}/{}",
      shard_id, shard_spec.name, shard_spec.dims.size(), shard_num_elements,
      registered_data.registered_size, registered_data.expected_size);

  return shard_metadata;
}
//==============================================================================
bool MetaStore::ValidateShardRegistration(
    const TensorShardSpec& shard_spec,
    const RegisteredShardsData& registered_data) const {
  // First shard for this tensor - nothing to validate against
  if (registered_data.shards.empty()) {
    return true;
  }

  // Get reference shard for metadata comparison
  const auto& ref_spec = registered_data.shards.begin()->second->spec;

  // Check dtype matches
  if (shard_spec.dtype != ref_spec.dtype) {
    LOG_WARNING("Dtype mismatch for tensor '{}': new shard has {}, expected {}",
                shard_spec.name, shard_spec.dtype, ref_spec.dtype);
    return false;
  }

  // Check dimension count matches
  if (shard_spec.dims.size() != ref_spec.dims.size()) {
    LOG_WARNING(
        "Dimension count mismatch for tensor '{}': new shard has {}, expected "
        "{}",
        shard_spec.name, shard_spec.dims.size(), ref_spec.dims.size());
    return false;
  }

  // Check dimension names and sizes match
  auto [shard_it, ref_it] = std::ranges::mismatch(
      shard_spec.dims, ref_spec.dims, [](const auto& a, const auto& b) {
        return a.name == b.name && a.size == b.size;
      });

  if (shard_it != shard_spec.dims.end()) {
    auto idx = std::distance(shard_spec.dims.begin(), shard_it);
    LOG_WARNING(
        "Dimension mismatch for tensor '{}' at index {}: "
        "provided (name={}, size={}) vs expected (name={}, size={})",
        shard_spec.name, idx, shard_it->name, shard_it->size, ref_it->name,
        ref_it->size);
    return false;
  }

  // Check for overlaps with all existing shards
  auto overlap_it = std::ranges::find_if(
      registered_data.shards, [&shard_spec](const auto& pair) {
        return shard_spec.Overlaps(pair.second->spec);
      });

  if (overlap_it != registered_data.shards.end()) {
    LOG_WARNING("Shard for tensor '{}' overlaps with existing shard {}",
                shard_spec.name, overlap_it->first);
    return false;
  }

  return true;
}
//==============================================================================
bool MetaStore::AllShardsRegistered(const TensorName& tensor_name) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = registered_shards_data_.find(tensor_name);
  if (it == registered_shards_data_.end()) {
    return false;
  }
  return it->second.registered_size == it->second.expected_size;
}
//==============================================================================
std::size_t MetaStore::GetNumShardsForTensor(
    const TensorName& tensor_name) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = registered_shards_data_.find(tensor_name);
  if (it != registered_shards_data_.end()) {
    return it->second.shards.size();
  }
  return 0;
}
//==============================================================================
TensorMetadataPtr MetaStore::GetTensorMetadata(const TensorName& tensor_name) {
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

  auto tensor_it = registered_shards_data_.find(tensor_name);
  ASSERT_VALID_RUNTIME(tensor_it != registered_shards_data_.end(),
                       "Tensor {} should exist if all shards are registered",
                       tensor_name);

  const auto& tensor_data = tensor_it->second;
  const auto& shards = tensor_data.shards;
  ASSERT_VALID_RUNTIME(!shards.empty(),
                       "Tensor {} should have at least one shard", tensor_name);

  // Get dims and dtype from first shard
  const auto& first_shard = shards.begin()->second;
  TensorDimMap dims;
  for (const auto& dim_spec : first_shard->spec.dims) {
    dims.emplace(dim_spec.name, TensorDim(dim_spec.name, dim_spec.size));
  }

  // Build and cache TensorMetadata
  auto metadata = std::make_shared<TensorMetadata>(
      tensor_name, dims, first_shard->spec.dtype, shards);
  tensor_metadata_cache_.emplace(tensor_name, metadata);

  return metadata;
}
//==============================================================================
const TensorSpec* MetaStore::GetTensorSpec(
    const TensorName& tensor_name) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = tensor_spec_cache_.find(tensor_name);
  if (it != tensor_spec_cache_.end()) {
    return &it->second;
  }
  return nullptr;
}
//==============================================================================
bool MetaStore::IsTensorDeregistered(const TensorName& tensor_name) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = registered_shards_data_.find(tensor_name);
  if (it == registered_shards_data_.end()) {
    return false;
  }
  return it->second.has_deregistered_shards;
}
void MetaStore::MarkTensorDeregistered(const TensorName& tensor_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = registered_shards_data_.find(tensor_name);
  if (it == registered_shards_data_.end()) {
    LOG_WARNING("MarkTensorDeregistered: tensor '{}' not found", tensor_name);
    return;
  }
  it->second.has_deregistered_shards = true;
}
//==============================================================================
void MetaStore::DeregisterShards(
    const std::unordered_map<TensorName, std::vector<ShardId>>&
        shards_by_tensor) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  for (const auto& [tensor_name, shard_ids] : shards_by_tensor) {
    auto it = registered_shards_data_.find(tensor_name);
    if (it == registered_shards_data_.end()) {
      LOG_WARNING("DeregisterShards: tensor '{}' not found", tensor_name);
      continue;
    }
    auto& data = it->second;
    data.has_deregistered_shards = true;

    for (const auto& shard_id : shard_ids) {
      auto shard_it = data.shards.find(shard_id);
      if (shard_it == data.shards.end()) {
        LOG_WARNING("DeregisterShards: shard {} not found in tensor '{}'",
                    shard_id, tensor_name);
        continue;
      }

      std::size_t shard_num_elements = shard_it->second->spec.GetNumElements();
      data.registered_size -= shard_num_elements;
      data.shards.erase(shard_it);

      LOG_DEBUG("Deregistered shard {} from tensor '{}', registered={}/{}",
                shard_id, tensor_name, data.registered_size,
                data.expected_size);
    }

    // Invalidate cached TensorMetadata for this tensor
    tensor_metadata_cache_.erase(tensor_name);

    // If no shards remain, clean up the tensor entirely
    if (data.shards.empty()) {
      LOG_DEBUG("All shards removed for tensor '{}', cleaning up", tensor_name);
      registered_shards_data_.erase(it);
      tensor_spec_cache_.erase(tensor_name);
    }
  }
}
//==============================================================================
}  // namespace setu::metastore
//==============================================================================
