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
//==============================================================================
namespace setu::commons::enums {
//==============================================================================
// Device abstraction
enum class DeviceKind : std::uint8_t { kCuda = 0, kCpu = 1, kNvme = 2 };
//==============================================================================
enum class ErrorCode : std::uint32_t {
  kSuccess = 0,
  kInvalidArguments = 1,
  kTimeout = 2,
  kInternalError = 3,
  kTensorNotFound = 4
};
//==============================================================================
}  // namespace setu::commons::enums
//==============================================================================
// std::formatter specializations for enums
//==============================================================================
/// @brief Formatter specialization for DeviceKind enum
template <>
struct std::formatter<setu::commons::enums::DeviceKind>
    : std::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(setu::commons::enums::DeviceKind type, FormatContext& ctx) const {
    std::string_view name = "UNKNOWN";
    switch (type) {
      case setu::commons::enums::DeviceKind::kCuda:
        name = "CUDA";
        break;
      case setu::commons::enums::DeviceKind::kCpu:
        name = "CPU";
        break;
      case setu::commons::enums::DeviceKind::kNvme:
        name = "NVME";
        break;
    }
    return std::formatter<std::string_view>::format(name, ctx);
  }
};
//==============================================================================
template <>
struct std::formatter<setu::commons::enums::ErrorCode>
    : std::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(setu::commons::enums::ErrorCode type, FormatContext& ctx) const {
    std::string_view name = "UNKNOWN";
    switch (type) {
      case setu::commons::enums::ErrorCode::kSuccess:
        name = "SUCCESS";
        break;
      case setu::commons::enums::ErrorCode::kInvalidArguments:
        name = "INVALID_ARGUMENTS";
        break;
      case setu::commons::enums::ErrorCode::kTimeout:
        name = "TIMEOUT";
        break;
      case setu::commons::enums::ErrorCode::kInternalError:
        name = "INTERNAL_ERROR";
        break;
      case setu::commons::enums::ErrorCode::kTensorNotFound:
        name = "TENSOR_NOT_FOUND";
        break;
    }
    return std::formatter<std::string_view>::format(name, ctx);
  }
};
//==============================================================================
