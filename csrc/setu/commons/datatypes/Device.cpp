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
#include "commons/datatypes/Device.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================
void Device::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  // Serialize torch::Device as device_type (int8) + device_index (int16)
  auto device_type = static_cast<std::int8_t>(torch_device.type());
  auto device_index = static_cast<std::int16_t>(torch_device.index());
  writer.WriteFields(device_type, device_index);
}

Device Device::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [device_type_val, device_index_val] =
      reader.ReadFields<std::int8_t, std::int16_t>();
  auto torch_device = torch::Device(
      static_cast<c10::DeviceType>(device_type_val), device_index_val);
  return Device(torch_device);
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
