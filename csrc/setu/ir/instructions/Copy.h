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
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::commons::DevicePtr;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorShardIdentifier;
//==============================================================================

struct CopyInstruction {
  CopyInstruction(TensorShardIdentifier src_tensor,
                  std::size_t src_memory_offset_bytes,
                  TensorShardIdentifier dst_tensor,
                  std::size_t dst_memory_offset_bytes, torch::Dtype dtype,
                  std::size_t num_elements, DevicePtr src_ptr = nullptr,
                  DevicePtr dst_ptr = nullptr)
      : src_tensor(std::move(src_tensor)),
        src_memory_offset_bytes(src_memory_offset_bytes),
        dst_tensor(std::move(dst_tensor)),
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
   * @brief Populates the device pointers by looking up the base address
   * @param resolver A callable that takes a TensorShardIdentifier and returns
   * the base DevicePtr.
   */
  void Embellish(
      const std::function<DevicePtr(const TensorShardIdentifier&)>& resolver);

  TensorShardIdentifier src_tensor;
  std::size_t src_memory_offset_bytes;
  TensorShardIdentifier dst_tensor;
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
