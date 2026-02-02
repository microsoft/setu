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
namespace setu::ir {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorShardIdentifier;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

struct SendInstruction {
  SendInstruction(DeviceRank dst_device_id, TensorShardIdentifier src_tensor,
                  torch::Dtype dtype, std::size_t memory_offset_bytes,
                  std::size_t num_elements, DevicePtr src_ptr = nullptr)
      : dst_device_id(dst_device_id),
        src_tensor(std::move(src_tensor)),
        dtype(dtype),
        memory_offset_bytes(memory_offset_bytes),
        num_elements(num_elements),
        src_ptr(src_ptr) {}

  ~SendInstruction() = default;
  SendInstruction(const SendInstruction&) = default;
  SendInstruction& operator=(const SendInstruction&) = default;
  SendInstruction(SendInstruction&&) = default;
  SendInstruction& operator=(SendInstruction&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static SendInstruction Deserialize(const BinaryRange& range);

  /**
   * @brief Populates the device pointers by looking up the base address
   * @param resolver A callable that takes a TensorShardIdentifier and returns
   * the base DevicePtr.
   */
  void Embellish(
      const std::function<DevicePtr(const TensorShardIdentifier&)>& resolver);

  DeviceRank dst_device_id;
  TensorShardIdentifier src_tensor;
  torch::Dtype dtype;
  std::size_t memory_offset_bytes;
  std::size_t num_elements;

  // Embellished pointers
  DevicePtr src_ptr;
};

//==============================================================================
}  // namespace setu::ir
//==============================================================================
