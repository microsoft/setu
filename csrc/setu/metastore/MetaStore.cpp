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
#include "coordinator/metastore/MetaStore.h"
//==============================================================================
#include "commons/BoostCommon.h"
#include "commons/datatypes/TensorDim.h"
//==============================================================================
namespace setu::coordinator::metastore {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::datatypes::TensorDim;
using setu::commons::datatypes::TensorDimMap;
//==============================================================================
TensorShardRef MetaStore::RegisterTensorShard(
    const TensorShardSpec& shard_spec) {
  // Generate a unique shard ID
  ShardId shard_id = GenerateUUID();

  // Convert TensorDimSpec vector to TensorDimMap
  TensorDimMap dims;
  for (const auto& dim_spec : shard_spec.dims) {
    dims.emplace(dim_spec.name, TensorDim(dim_spec.name, dim_spec.size));
  }

  // Create the shard reference
  TensorShardRef shard_ref(shard_spec.name, shard_id, dims);

  shards_by_id_.emplace(shard_id, shard_ref);
  shards_by_tensor_name_[shard_spec.name].push_back(shard_id);

  // Calculate and track sizes
  std::size_t shard_num_elements = shard_spec.GetNumElements();
  std::size_t total_tensor_size = 1;
  for (const auto& dim_spec : shard_spec.dims) {
    total_tensor_size *= dim_spec.size;
  }

  if (tensor_expected_size_.find(shard_spec.name) ==
      tensor_expected_size_.end()) {
    tensor_expected_size_[shard_spec.name] = total_tensor_size;
    tensor_registered_size_[shard_spec.name] = 0;
  }

  tensor_registered_size_[shard_spec.name] += shard_num_elements;

  LOG_DEBUG(
      "Registered tensor shard: id={}, name={}, num_dims={}, "
      "shard_elements={}, registered={}/{}",
      shard_id, shard_spec.name, dims.size(), shard_num_elements,
      tensor_registered_size_[shard_spec.name],
      tensor_expected_size_[shard_spec.name]);

  return shard_ref;
}
//==============================================================================
std::optional<TensorShardRef> MetaStore::GetShardById(
    const ShardId& shard_id) const {
  auto it = shards_by_id_.find(shard_id);
  if (it != shards_by_id_.end()) {
    return it->second;
  }
  return std::nullopt;
}
//==============================================================================
std::vector<TensorShardRef> MetaStore::GetShardsByTensorName(
    const TensorName& tensor_name) const {
  std::vector<TensorShardRef> result;

  auto it = shards_by_tensor_name_.find(tensor_name);
  if (it != shards_by_tensor_name_.end()) {
    for (const auto& shard_id : it->second) {
      auto shard_it = shards_by_id_.find(shard_id);
      if (shard_it != shards_by_id_.end()) {
        result.push_back(shard_it->second);
      }
    }
  }

  return result;
}
//==============================================================================
std::size_t MetaStore::GetNumShards() const { return shards_by_id_.size(); }
//==============================================================================
bool MetaStore::AllShardsRegistered(const TensorName& tensor_name) const {
  auto expected_it = tensor_expected_size_.find(tensor_name);
  auto registered_it = tensor_registered_size_.find(tensor_name);

  if (expected_it == tensor_expected_size_.end() ||
      registered_it == tensor_registered_size_.end()) {
    return false;
  }

  return registered_it->second == expected_it->second;
}
//==============================================================================
std::size_t MetaStore::GetNumShardsForTensor(
    const TensorName& tensor_name) const {
  auto it = shards_by_tensor_name_.find(tensor_name);
  if (it != shards_by_tensor_name_.end()) {
    return it->second.size();
  }
  return 0;
}
//==============================================================================
}  // namespace setu::coordinator::metastore
//==============================================================================