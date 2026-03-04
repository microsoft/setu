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
#include "planner/ir/llc/ShardAccessTypes.h"
#include "planner/ir/ref/BufferRef.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::planner::ir::ref::BufferRef;
using setu::planner::ir::ref::ShardRef;
//==============================================================================

/// A single copy entry within a batched Copy instruction.
struct CopyEntry {
  CopyEntry(BufferRef src_ref_param, std::size_t src_offset_bytes_param,
            BufferRef dst_ref_param, std::size_t dst_offset_bytes_param,
            std::size_t count_param, torch::Dtype dtype_param,
            DevicePtr src_ptr_param = nullptr,
            DevicePtr dst_ptr_param = nullptr)
      : src_ref(std::move(src_ref_param)),
        src_offset_bytes(src_offset_bytes_param),
        dst_ref(std::move(dst_ref_param)),
        dst_offset_bytes(dst_offset_bytes_param),
        count(count_param),
        dtype(dtype_param),
        src_ptr(src_ptr_param),
        dst_ptr(dst_ptr_param) {}

  ~CopyEntry() = default;
  CopyEntry(const CopyEntry&) = default;
  CopyEntry& operator=(const CopyEntry&) = default;
  CopyEntry(CopyEntry&&) = default;
  CopyEntry& operator=(CopyEntry&&) = default;

  BufferRef src_ref;
  std::size_t src_offset_bytes;
  BufferRef dst_ref;
  std::size_t dst_offset_bytes;
  std::size_t count;
  torch::Dtype dtype;

  // Embellished pointers — resolved lazily via Copy::Embellish().
  DevicePtr src_ptr;
  DevicePtr dst_ptr;
};

/// Batched local (same-device) memory copy instruction.
///
/// Contains one or more CopyEntry items, each describing a copy of `count`
/// elements of type `dtype` from `src_ref` at `src_offset_bytes` to `dst_ref`
/// at `dst_offset_bytes`.  All entries must target the same device.
///
/// At execution time the worker issues a single `cudaMemcpyBatchAsync` call
/// for all entries, reducing per-copy driver overhead.  Device pointers are
/// resolved lazily via Embellish() before execution.
struct Copy {
  explicit Copy(std::vector<CopyEntry> entries_param)
      : entries(std::move(entries_param)) {}

  /// Convenience constructor for a single-entry copy.
  Copy(BufferRef src_ref_param, std::size_t src_offset_bytes_param,
       BufferRef dst_ref_param, std::size_t dst_offset_bytes_param,
       std::size_t count_param, torch::Dtype dtype_param) {
    entries.emplace_back(std::move(src_ref_param), src_offset_bytes_param,
                         std::move(dst_ref_param), dst_offset_bytes_param,
                         count_param, dtype_param);
  }

  ~Copy() = default;
  Copy(const Copy&) = default;
  Copy& operator=(const Copy&) = default;
  Copy(Copy&&) = default;
  Copy& operator=(Copy&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static Copy Deserialize(const BinaryRange& range);

  /// Populates device pointers for all entries by looking up base addresses.
  void Embellish(const std::function<DevicePtr(const BufferRef&)>& resolver);

  /// @brief Extract shard access requirements for this instruction.
  [[nodiscard]] ShardAccessMap GetShardAccess() const;

  std::vector<CopyEntry> entries;
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
