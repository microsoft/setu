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
#include "commons/datatypes/DeviceId.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
DeviceId::DeviceId(std::string value_param) : value_(std::move(value_param)) {
  ASSERT_VALID_ARGUMENTS(!value_.empty(), "DeviceId value must be non-empty");
}
//==============================================================================
void DeviceId::Serialize(BinaryBuffer& buffer) const {
  setu::commons::utils::BinaryWriter writer(buffer);
  writer.WriteFields(value_);
}
//==============================================================================
DeviceId DeviceId::Deserialize(const BinaryRange& range) {
  setu::commons::utils::BinaryReader reader(range);
  auto [value_val] = reader.ReadFields<std::string>();
  DeviceId id;
  id.value_ = std::move(value_val);
  return id;
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
