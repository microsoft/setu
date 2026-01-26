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
#include "commons/datatypes/TensorShard.h"

#include "commons/TorchCommon.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================
void TensorShard::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  // Serialize device_ptr as a raw address value (valid only intra-process).
  const auto device_ptr_value = reinterpret_cast<std::uintptr_t>(device_ptr);
  writer.WriteFields(id, name, device, device_ptr_value, dtype, dim_shards);
}

TensorShard TensorShard::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [id_val, name_val, device_val, device_ptr_value, dtype_val,
        dim_shards_val] =
      reader.ReadFields<ShardId, TensorName, Device, std::uintptr_t,
                        torch::Dtype, TensorDimShardsMap>();
  auto device_ptr_val = reinterpret_cast<DevicePtr>(device_ptr_value);
  return TensorShard(id_val, name_val, device_val, device_ptr_val, dtype_val,
                     dim_shards_val);
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
