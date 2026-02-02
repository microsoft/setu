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
#include "setu/ir/ShardRef.h"
//==============================================================================
namespace setu::ir {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

struct CopyInstruction {
  CopyInstruction(ShardRef src_shard, std::size_t src_memory_offset_bytes,
                  ShardRef dst_shard, std::size_t dst_memory_offset_bytes,
                  torch::Dtype dtype, std::size_t num_elements,
                  DevicePtr src_ptr = nullptr, DevicePtr dst_ptr = nullptr)
      : src_shard(std::move(src_shard)),
        src_memory_offset_bytes(src_memory_offset_bytes),
        dst_shard(std::move(dst_shard)),
        dst_memory_offset_bytes(dst_memory_offset_bytes),
        dtype(dtype),
        num_elements(num_elements),
        src_ptr{src_ptr},
        dst_ptr{dst_ptr} {}

  ~CopyInstruction() = default;
  CopyInstruction(const CopyInstruction&) = default;
  CopyInstruction& operator=(const CopyInstruction&) = default;
  CopyInstruction(CopyInstruction&&) = default;
  CopyInstruction& operator=(CopyInstruction&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static CopyInstruction Deserialize(const BinaryRange& range);

  /**
   * @brief Populates the device pointers by looking up the base address.
   */
  void Embellish(const std::function<DevicePtr(const ShardRef&)>& resolver);

  ShardRef src_shard;
  std::size_t src_memory_offset_bytes;
  ShardRef dst_shard;
  std::size_t dst_memory_offset_bytes;
  torch::Dtype dtype;
  std::size_t num_elements;

  // Embellished pointers
  DevicePtr src_ptr;
  DevicePtr dst_ptr;
};

//==============================================================================
}  // namespace setu::ir
//==============================================================================
