//==============================================================================
// Copyright 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation
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
//==============================================================================
#include "commons/ClassTraits.h"
#include "commons/Logging.h"
#include "commons/Types.h"
#include "commons/datatypes/TensorShard.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief RAII handle for read-only access to tensor shard device memory
 *
 * Provides thread-safe read access to a tensor shard's device memory using
 * a shared lock. Multiple readers can hold read handles simultaneously.
 */
class TensorShardReadHandle : public NonCopyableNonMovable {
 public:
  /**
   * @brief Constructs a read handle and acquires shared lock
   *
   * @param shard_param Tensor shard to acquire read access for
   *
   * @throws std::invalid_argument if shard is null
   */
  explicit TensorShardReadHandle(TensorShardPtr shard_param)
      : shard(shard_param), lock(shard->mutex) {
    ASSERT_VALID_POINTER_ARGUMENT(shard_param);
    LOG_DEBUG("Acquired read lock for shard: {}", shard->name);
  }

  /**
   * @brief Releases the shared lock on destruction
   */
  ~TensorShardReadHandle() {
    LOG_DEBUG("Released read lock for shard: {}", shard->name);
  }

  /**
   * @brief Get read-only pointer to device memory
   *
   * @return Const pointer to device memory
   */
  [[nodiscard]] DevicePtr GetDevicePtr() const { return shard->device_ptr; }

  /**
   * @brief Get the shard being accessed
   *
   * @return Const pointer to the tensor shard
   */
  [[nodiscard]] const TensorShard* GetShard() const { return shard.get(); }

 private:
  const TensorShardPtr shard;                ///< Shard being accessed
  std::shared_lock<std::shared_mutex> lock;  ///< Shared lock for read access
};
//==============================================================================
/**
 * @brief RAII handle for read-write access to tensor shard device memory
 *
 * Provides thread-safe write access to a tensor shard's device memory using
 * an exclusive lock. Only one writer can hold a write handle at a time, and
 * no readers can access while a write lock is held.
 */
class TensorShardWriteHandle : public NonCopyableNonMovable {
 public:
  /**
   * @brief Constructs a write handle and acquires exclusive lock
   *
   * @param shard_param Tensor shard to acquire write access for
   *
   * @throws std::invalid_argument if shard is null
   */
  explicit TensorShardWriteHandle(TensorShardPtr shard_param)
      : shard(shard_param), lock(shard->mutex) {
    ASSERT_VALID_POINTER_ARGUMENT(shard_param);
    LOG_DEBUG("Acquired write lock for shard: {}", shard->name);
  }

  /**
   * @brief Releases the exclusive lock on destruction
   */
  ~TensorShardWriteHandle() {
    LOG_DEBUG("Released write lock for shard: {}", shard->name);
  }

  /**
   * @brief Get read-write pointer to device memory
   *
   * @return Pointer to device memory
   */
  [[nodiscard]] DevicePtr GetDevicePtr() const { return shard->device_ptr; }

  /**
   * @brief Get the shard being accessed
   *
   * @return Pointer to the tensor shard
   */
  [[nodiscard]] TensorShard* GetShard() const { return shard.get(); }

 private:
  const TensorShardPtr shard;  ///< Shard being accessed
  std::unique_lock<std::shared_mutex>
      lock;  ///< Exclusive lock for write access
};
//==============================================================================
/// @brief Shared pointer to TensorShardReadHandle
using TensorShardReadHandlePtr = std::shared_ptr<TensorShardReadHandle>;

/// @brief Shared pointer to TensorShardWriteHandle
using TensorShardWriteHandlePtr = std::shared_ptr<TensorShardWriteHandle>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
