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
#include "setu/ir/instructions/Receive.h"
//==============================================================================
namespace setu::ir {
//==============================================================================

std::string ReceiveInstruction::ToString() const {
  return std::format(
      "ReceiveInstruction(src_rank={}, tensor=({}, {}), dtype={}, "
      "memory_offset={}, num_elements={}, dst_device_ptr = {})",
      src_device_id, dst_tensor.tensor_name, dst_tensor.shard_id,
      static_cast<int>(dtype), memory_offset_bytes, num_elements, dst_ptr);
}

void ReceiveInstruction::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  const auto dst_ptr_value = reinterpret_cast<std::uintptr_t>(dst_ptr);
  writer.WriteFields(src_device_id, dst_tensor.tensor_name, dst_tensor.shard_id,
                     dtype, memory_offset_bytes, num_elements, dst_ptr_value);
}

ReceiveInstruction ReceiveInstruction::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [src_device_id, tensor_name, shard_id, dtype, memory_offset_bytes,
        num_elements, dst_ptr_value] =
      reader.ReadFields<DeviceRank, TensorName, ShardId, torch::Dtype,
                        std::size_t, std::size_t, std::uintptr_t>();
  const auto dst_ptr = reinterpret_cast<DevicePtr>(dst_ptr_value);
  return ReceiveInstruction(src_device_id,
                            {std::move(tensor_name), std::move(shard_id)}, dtype,
                            memory_offset_bytes, num_elements, dst_ptr);
}

void ReceiveInstruction::Embellish(
    const std::function<DevicePtr(const TensorShardIdentifier&)>& resolver) {
  dst_ptr = resolver(dst_tensor);
}

//==============================================================================
}  // namespace setu::ir
//==============================================================================
