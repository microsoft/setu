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
namespace setu::commons {
//==============================================================================
enum class LogLevel {
  kDebug = 0,
  kInfo = 1,
  kWarning = 2,
  kError = 3,
  kCritical = 4
};
//==============================================================================
}  // namespace setu::commons
//==============================================================================
/// @brief Formatter specialization for LogLevel enum
template <>
struct std::formatter<setu::commons::LogLevel>
    : std::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(setu::commons::LogLevel level, FormatContext& ctx) const {
    std::string_view name = "UNKNOWN";
    switch (level) {
      case setu::commons::LogLevel::kDebug:
        name = "DEBUG";
        break;
      case setu::commons::LogLevel::kInfo:
        name = "INFO";
        break;
      case setu::commons::LogLevel::kWarning:
        name = "WARNING";
        break;
      case setu::commons::LogLevel::kError:
        name = "ERROR";
        break;
      case setu::commons::LogLevel::kCritical:
        name = "CRITICAL";
        break;
    }
    return std::formatter<std::string_view>::format(name, ctx);
  }
};
//==============================================================================
