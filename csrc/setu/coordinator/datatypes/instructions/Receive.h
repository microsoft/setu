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
#include "setu/commons/StdCommon.h"
#include "setu/commons/Types.h"
#include "setu/commons/datatypes/TensorShardIdentifier.h"
#include "setu/commons/enums/Enums.h"
#include "setu/commons/utils/Serialization.h"
//==============================================================================
namespace setu::coordinator::datatypes::instructions {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorShardIdentifier;
//==============================================================================

struct ReceiveInstruction {
  ReceiveInstruction(DeviceRank src_device_id,
                     TensorShardIdentifier dst_tensor,
                     torch::Dtype dtype,
                     std::size_t memory_offset_bytes,
                     std::size_t num_elements,
                     DevicePtr dst_ptr=nullptr)
      : src_device_id(src_device_id),
        dst_tensor(std::move(dst_tensor)),
        dtype(dtype),
        memory_offset_bytes(memory_offset_bytes),
        num_elements(num_elements),
        dst_ptr(dst_ptr) {}

  ~ReceiveInstruction() = default;
  ReceiveInstruction(const ReceiveInstruction&) = default;
  ReceiveInstruction& operator=(const ReceiveInstruction&) = default;
  ReceiveInstruction(ReceiveInstruction&&) = default;
  ReceiveInstruction& operator=(ReceiveInstruction&&) = default;

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "ReceiveInstruction(src_rank={}, tensor=({}, {}), dtype={}, "
        "memory_offset={}, num_elements={}, dst_device_ptr = {})",
        src_device_id, dst_tensor.tensor_name, dst_tensor.shard_id,
        static_cast<int>(dtype), memory_offset_bytes, num_elements, dst_ptr);
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    const auto dst_ptr_value = reinterpret_cast<std::uintptr_t>(dst_ptr);
    writer.WriteFields(src_device_id, dst_tensor.tensor_name, dst_tensor.shard_id,
                       dtype, memory_offset_bytes, num_elements, dst_ptr_value);
  }

  static ReceiveInstruction Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto [src_device_id, tensor_name, shard_id, dtype, memory_offset_bytes,
          num_elements, dst_ptr_value] =
        reader.ReadFields<DeviceRank, TensorName, ShardId, torch::Dtype, std::size_t,
                          std::size_t, std::uintptr_t>();
    const auto dst_ptr = reinterpret_cast<DevicePtr>(dst_ptr_value);
    return ReceiveInstruction(src_device_id,
                              {std::move(tensor_name), std::move(shard_id)},
                              dtype, memory_offset_bytes, num_elements, dst_ptr);
  }

  /**
   * @brief Populates the device pointers by looking up the base address
   * @param resolver A callable that takes (TensorName, ShardId) and returns the base DevicePtr.
   */
  void Embellish(
      const std::function<DevicePtr(const TensorName&, const ShardId&)>& resolver) {
    dst_ptr = resolver(dst_tensor.tensor_name, dst_tensor.shard_id);
  }

  DeviceRank src_device_id;
  TensorShardIdentifier dst_tensor;
  torch::Dtype dtype;
  std::size_t memory_offset_bytes;
  std::size_t num_elements;

  // Embellished pointers
  DevicePtr dst_ptr;
};

//==============================================================================
}  // namespace setu::coordinator::datatypes::instructions
//==============================================================================
