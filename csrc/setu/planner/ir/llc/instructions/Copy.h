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
#include "commons/enums/Enums.h"
#include "commons/utils/Serialization.h"
//==============================================================================
#include "planner/ir/llc/ShardRef.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

/// Local (same-device) memory copy between two shard regions.
///
/// Copies `count` elements of type `dtype` from `src_shard` at
/// `src_offset_bytes` to `dst_shard` at `dst_offset_bytes`.  Both shards
/// must reside on the same device.  Device pointers are resolved lazily via
/// Embellish() before execution.
struct Copy {
  Copy(ShardRef src_shard_param, std::size_t src_offset_bytes_param,
       ShardRef dst_shard_param, std::size_t dst_offset_bytes_param,
       std::size_t count_param, torch::Dtype dtype_param,
       DevicePtr src_ptr_param = nullptr, DevicePtr dst_ptr_param = nullptr)
      : src_shard(std::move(src_shard_param)),
        src_offset_bytes(src_offset_bytes_param),
        dst_shard(std::move(dst_shard_param)),
        dst_offset_bytes(dst_offset_bytes_param),
        count(count_param),
        dtype(dtype_param),
        src_ptr(src_ptr_param),
        dst_ptr(dst_ptr_param) {}

  ~Copy() = default;
  Copy(const Copy&) = default;
  Copy& operator=(const Copy&) = default;
  Copy(Copy&&) = default;
  Copy& operator=(Copy&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static Copy Deserialize(const BinaryRange& range);

  /**
   * @brief Populates the device pointers by looking up the base address.
   */
  void Embellish(const std::function<DevicePtr(const ShardRef&)>& resolver);

  ShardRef src_shard;
  std::size_t src_offset_bytes;
  ShardRef dst_shard;
  std::size_t dst_offset_bytes;
  std::size_t count;
  torch::Dtype dtype;

  // Embellished pointers
  DevicePtr src_ptr;
  DevicePtr dst_ptr;
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
