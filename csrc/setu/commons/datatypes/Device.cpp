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
#include "commons/utils/DeviceMap.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::commons::utils::DeviceMap;
//==============================================================================
Device::Device(torch::Device torch_device_param)
    : torch_device(torch_device_param) {
  if (torch_device.type() == torch::kCUDA && torch_device.has_index()) {
    const auto& map = DeviceMap::Local();
    if (map.Contains(static_cast<int>(torch_device.index()))) {
      device_id = map.GetDeviceId(torch_device.index());
    }
  }
  ASSERT_VALID_RUNTIME(
      !device_id.Empty(),
      "Device must have a canonical device_id; DeviceMap::Local() has no "
      "mapping for torch_device={}",
      torch_device.str());
}
//==============================================================================
Device::Device(DeviceId device_id_param)
    : device_id(std::move(device_id_param)) {
  ASSERT_VALID_ARGUMENTS(!device_id.Empty(),
                         "Device must be constructed with a non-empty "
                         "device_id");
  // Without a mapping for device_id, torch_device cannot be resolved in this
  // process; leave it at the default-constructed sentinel.
  const auto& map = DeviceMap::Local();
  if (map.Contains(device_id)) {
    torch_device = torch::Device(
        torch::kCUDA,
        static_cast<torch::DeviceIndex>(map.GetLocalIndex(device_id)));
    // Only CUDA devices are supported at the moment.
  }
}
//==============================================================================
void Device::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  auto device_type = static_cast<std::int8_t>(torch_device.type());
  writer.WriteFields(device_type, device_id);
}
//==============================================================================
Device Device::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [device_type_val, device_id_val] =
      reader.ReadFields<std::int8_t, DeviceId>();
  auto device_type = static_cast<c10::DeviceType>(device_type_val);
  if (device_type == torch::kCUDA && !device_id_val.Empty()) {
    return Device(std::move(device_id_val));
  }
  Device d;
  d.torch_device = torch::Device(device_type);
  d.device_id = std::move(device_id_val);
  return d;
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
