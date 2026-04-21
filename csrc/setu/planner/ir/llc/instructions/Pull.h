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
#include "commons/datatypes/DeviceId.h"
#include "commons/enums/Enums.h"
#include "commons/utils/Serialization.h"
//==============================================================================
#include "planner/ir/llc/ShardAccessTypes.h"
#include "planner/ir/ref/BufferRef.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::datatypes::DeviceId;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::planner::ir::ref::BufferRef;
using setu::planner::ir::ref::ShardRef;
//==============================================================================

/// A single entry within a batched Pull instruction.
struct PullEntry {
  PullEntry(BufferRef src_ref_param, std::size_t src_offset_bytes_param,
            BufferRef dst_ref_param, std::size_t dst_offset_bytes_param,
            std::size_t count_param, torch::Dtype dtype_param,
            DeviceId src_device_param, DevicePtr src_ptr_param = nullptr,
            DevicePtr dst_ptr_param = nullptr)
      : src_ref(std::move(src_ref_param)),
        src_offset_bytes(src_offset_bytes_param),
        dst_ref(std::move(dst_ref_param)),
        dst_offset_bytes(dst_offset_bytes_param),
        count(count_param),
        dtype(dtype_param),
        src_device(std::move(src_device_param)),
        src_ptr(src_ptr_param),
        dst_ptr(dst_ptr_param) {}

  ~PullEntry() = default;
  PullEntry(const PullEntry&) = default;
  PullEntry& operator=(const PullEntry&) = default;
  PullEntry(PullEntry&&) = default;
  PullEntry& operator=(PullEntry&&) = default;

  BufferRef src_ref;
  std::size_t src_offset_bytes;
  BufferRef dst_ref;
  std::size_t dst_offset_bytes;
  std::size_t count;
  torch::Dtype dtype;
  DeviceId src_device;

  // Embellished pointers
  DevicePtr src_ptr;
  DevicePtr dst_ptr;
};

/// Batched P2P pull (receiver-side DMA) from remote devices on the same node.
///
/// Contains one or more PullEntry items. At execution time the worker issues
/// a single cudaMemcpyBatchAsync call for all entries. Follows UCX/RDMA
/// GET semantics.
struct Pull {
  explicit Pull(std::vector<PullEntry> entries_param)
      : entries(std::move(entries_param)) {}

  /// Convenience constructor for a single-entry pull.
  Pull(BufferRef src_ref_param, std::size_t src_offset_bytes_param,
       BufferRef dst_ref_param, std::size_t dst_offset_bytes_param,
       std::size_t count_param, torch::Dtype dtype_param,
       DeviceId src_device_param) {
    entries.emplace_back(std::move(src_ref_param), src_offset_bytes_param,
                         std::move(dst_ref_param), dst_offset_bytes_param,
                         count_param, dtype_param,
                         std::move(src_device_param));
  }

  ~Pull() = default;
  Pull(const Pull&) = default;
  Pull& operator=(const Pull&) = default;
  Pull(Pull&&) = default;
  Pull& operator=(Pull&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static Pull Deserialize(const BinaryRange& range);

  void Embellish(const std::function<DevicePtr(const BufferRef&)>& resolver);

  [[nodiscard]] ShardAccessMap GetShardAccess() const;

  std::vector<PullEntry> entries;
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
