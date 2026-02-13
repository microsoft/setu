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
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "commons/Types.h"
#include "commons/datatypes/TensorShardMetadata.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Represents a physical tensor shard with device memory
 *
 * TensorShard combines metadata about a shard with the physical device pointer
 * and a file-based read-write lock for inter-process synchronization.
 * Use TensorShardReadHandle and TensorShardWriteHandle for safe access to the
 * device memory.
 */
struct TensorShard {
  /**
   * @brief Constructs a tensor shard with metadata, tensor, and file-based lock
   *
   * Creates a lock file at a deterministic path derived from the shard's UUID
   * ({lock_base_dir}/{shard-uuid}.lock) and opens it for inter-process
   * synchronization.
   *
   * @param metadata_param Metadata describing this shard
   * @param tensor_param The torch tensor holding the shard data
   * @param lock_base_dir Directory where lock files are created
   *
   * @throws std::invalid_argument if tensor_param is not defined
   * @throws std::invalid_argument if lock_base_dir is empty
   */
  TensorShard(TensorShardMetadata metadata_param, torch::Tensor tensor_param,
              const std::string& lock_base_dir /*[in]*/)
      : metadata(std::move(metadata_param)),
        tensor(std::move(tensor_param)),
        lock_(CreateAndOpenFileLock(metadata.id, lock_base_dir)) {
    ASSERT_VALID_ARGUMENTS(tensor.defined() && tensor.numel() > 0,
                           "Invalid tensor argument: tensor is not defined");
  }

  /**
   * @brief Returns a string representation of the tensor shard
   *
   * @return String containing metadata and device pointer
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorShard(metadata={}, tensor={})",
                       metadata.ToString(), tensor);
  }

  /**
   * @brief Get read-only pointer to device memory
   *
   * @return Const pointer to device memory
   */
  [[nodiscard]] DevicePtr GetDevicePtr() const { return tensor.data_ptr(); }

  const TensorShardMetadata metadata;  ///< Immutable metadata for this shard
  const torch::Tensor tensor;          ///< The torch tensor holding shard data

 private:
  friend class TensorShardReadHandle;
  friend class TensorShardWriteHandle;

  /**
   * @brief Creates a lock file and opens a file lock for inter-process sync
   *
   * The lock file path is deterministic: {base_dir}/{shard-uuid}.lock.
   * Two processes using the same shard ID and base directory will share the
   * same lock file, enabling kernel-level reader-writer synchronization.
   *
   * @param shard_id The shard UUID used to derive the lock file name
   * @param base_dir Directory where the lock file is created
   *
   * @return Shared pointer to the opened file lock
   */
  [[nodiscard]] static FileLockPtr CreateAndOpenFileLock(
      const ShardId& shard_id /*[in]*/, const std::string& base_dir /*[in]*/) {
    ASSERT_VALID_ARGUMENTS(!base_dir.empty(),
                           "Lock base directory cannot be empty");

    std::error_code ec;
    std::filesystem::create_directories(base_dir, ec);
    ASSERT_VALID_RUNTIME(!ec, "Failed to create lock dir '{}': {}", base_dir,
                         ec.message());

    const auto lock_path = std::filesystem::path(base_dir) /
                           (boost::uuids::to_string(shard_id) + ".lock");

    // Touch file (create if missing; don't truncate if present)
    std::ofstream ofs(lock_path, std::ios::app);
    ASSERT_VALID_RUNTIME(ofs.good(), "Failed to create/open lock file '{}'",
                         lock_path.string());
    ofs.close();

    const auto abs_path = std::filesystem::absolute(lock_path, ec);
    ASSERT_VALID_RUNTIME(!ec, "Failed to make absolute path '{}': {}",
                         lock_path.string(), ec.message());

    LOG_DEBUG("Created lock file: {}", abs_path.string());
    return std::make_shared<FileLock>(abs_path.string().c_str());
  }

  const FileLockPtr lock_;
};
//==============================================================================
/// @brief Shared pointer to a TensorShard object
using TensorShardPtr = std::shared_ptr<TensorShard>;

/// @brief Map of shard IDs to TensorShard objects
using TensorShardsMap = std::unordered_map<ShardId, TensorShardPtr>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
