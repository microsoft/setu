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
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::BinaryBuffer;
using setu::commons::BinaryRange;
//==============================================================================
/**
 * @brief Canonical, process-invariant unique identifier for a physical device.
 *
 * DeviceId is the string the system carries across process boundaries to refer
 * to a specific physical device instance. The concrete format is defined by
 * whichever populator fills the process-local DeviceMap: for NVIDIA GPUs the
 * value is the NVML UUID (e.g. "GPU-441f01ed-4f8a-..."); other device classes
 * (AMD, TPU, etc.) will use their canonical equivalents when added.
 *
 * DeviceId itself is opaque to the format.
 */
struct DeviceId {
  DeviceId() = default;

  explicit DeviceId(std::string value_param /*[in]*/);

  [[nodiscard]] const std::string& ToString() const { return value_; }

  [[nodiscard]] bool Empty() const { return value_.empty(); }

  [[nodiscard]] bool operator==(const DeviceId& other) const {
    return value_ == other.value_;
  }
  [[nodiscard]] bool operator!=(const DeviceId& other) const {
    return !(*this == other);
  }
  [[nodiscard]] bool operator<(const DeviceId& other) const {
    return value_ < other.value_;
  }

  void Serialize(BinaryBuffer& buffer) const;

  static DeviceId Deserialize(const BinaryRange& range);

 private:
  std::string value_;
};
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
namespace std {
template <>
struct hash<setu::commons::datatypes::DeviceId> {
  std::size_t operator()(
      const setu::commons::datatypes::DeviceId& id) const noexcept {
    return std::hash<std::string>{}(id.ToString());
  }
};
}  // namespace std
//==============================================================================
