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
using setu::commons::DeviceRank;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

struct ReceiveInstruction {
  ReceiveInstruction(DeviceRank src_device_id, ShardRef dst_shard,
                     torch::Dtype dtype, std::size_t memory_offset_bytes,
                     std::size_t num_elements, DevicePtr dst_ptr = nullptr)
      : src_device_id(src_device_id),
        dst_shard(std::move(dst_shard)),
        dtype(dtype),
        memory_offset_bytes(memory_offset_bytes),
        num_elements(num_elements),
        dst_ptr(dst_ptr) {}

  ~ReceiveInstruction() = default;
  ReceiveInstruction(const ReceiveInstruction&) = default;
  ReceiveInstruction& operator=(const ReceiveInstruction&) = default;
  ReceiveInstruction(ReceiveInstruction&&) = default;
  ReceiveInstruction& operator=(ReceiveInstruction&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static ReceiveInstruction Deserialize(const BinaryRange& range);

  /**
   * @brief Populates the device pointers by looking up the base address.
   */
  void Embellish(const std::function<DevicePtr(const ShardRef&)>& resolver);

  DeviceRank src_device_id;
  ShardRef dst_shard;
  torch::Dtype dtype;
  std::size_t memory_offset_bytes;
  std::size_t num_elements;

  // Embellished pointers
  DevicePtr dst_ptr;
};

//==============================================================================
}  // namespace setu::ir
//==============================================================================
