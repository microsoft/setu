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
#include "commons/Types.h"
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/datatypes/TensorShardWrapper.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Represents a physical tensor shard with device memory
 *
 * TensorShard combines metadata about a shard with the physical device pointer
 * and thread-safe locking mechanisms. Use TensorShardReadHandle and
 * TensorShardWriteHandle for safe access to the device memory.
 */
struct TensorShard {
  /**
   * @brief Constructs a tensor shard with metadata and device pointer
   *
   * @param metadata_param Metadata describing this shard
   * @param tensor_shard_wrapper_param Setu tensor shard wrapper
   *
   * @throws std::invalid_argument if tensor_shard_wrapper_param is null
   */
  TensorShard(TensorShardMetadata metadata_param,
              TensorShardWrapper tensor_shard_wrapper_param)
      : metadata(std::move(metadata_param)),
        tensor_shard_wrapper(tensor_shard_wrapper_param) {}

  /**
   * @brief Returns a string representation of the tensor shard
   *
   * @return String containing metadata and device pointer
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorShard(metadata={}, tensor_shard_wrapper={})",
                       metadata.ToString(), tensor_shard_wrapper);
  }

  /**
   * @brief Get read-only pointer to device memory
   *
   * @return Const pointer to device memory
   */
  [[nodiscard]] DevicePtr GetDevicePtr() const {
    return tensor_shard_wrapper.GetDevicePtr();
  }

  /**
   * @brief Get internal tensor store
   * TODO: Add a method to cast to torch::Tensor
   * @return Const pointer to internal tensor store
   */
  [[nodiscard]] torch::Tensor GetTorchTensor() const {
    return tensor_shard_wrapper.GetTorchTensor();
  }

  const TensorShardMetadata metadata;  ///< Immutable metadata for this shard
  const TensorShardWrapper
      tensor_shard_wrapper;  ///< Pointer to setu tensor wrapper

 private:
  friend class TensorShardReadHandle;
  friend class TensorShardWriteHandle;
  mutable std::shared_mutex
      mutex;  ///< Mutex for thread-safe access to tensor_shard_wrapper
};
//==============================================================================
/// @brief Shared pointer to a TensorShard object
using TensorShardPtr = std::shared_ptr<TensorShard>;

/// @brief Map of shard IDs to TensorShard objects
using TensorShardsMap = std::unordered_map<ShardId, TensorShardPtr>;

/// @brief Lookup tensor shard given shard id
using TensorShardsConcurrentMap =
    ConcurrentMap<ShardId, TensorShardPtr>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
