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
namespace setu::commons::utils {
//==============================================================================

/// @brief Parse a typed value from a string.
template <typename T>
[[nodiscard]] T ParseValue(const std::string& s) {
  if constexpr (std::is_same_v<T, bool>) {
    return s == "1" || s == "true" || s == "TRUE";
  } else if constexpr (std::is_same_v<T, std::string>) {
    return s;
  } else if constexpr (std::is_unsigned_v<T>) {
    return static_cast<T>(std::stoull(s));
  } else if constexpr (std::is_signed_v<T>) {
    return static_cast<T>(std::stoll(s));
  }
}

/// @brief Read a typed value from an environment variable, or return a default.
///
/// Supported types: any integral type, bool, std::string.
///
/// Usage:
///   auto val = GetEnv<std::size_t>("SETU_WORKER_NUM_STREAMS", 2);
///   auto flag = GetEnv<bool>("SETU_ENABLE_DEBUG", false);
template <typename T>
[[nodiscard]] T GetEnv(const char* name /*[in]*/, T default_val /*[in]*/) {
  const char* env = std::getenv(name);
  if (env == nullptr) return default_val;
  return ParseValue<T>(std::string(env));
}

/// @brief Read a comma-separated list from an environment variable.
/// e.g. "1,2,3" -> std::vector<int>{1, 2, 3}
template <typename T>
[[nodiscard]] std::vector<T> GetEnv(const char* name /*[in]*/,
                                    std::vector<T> default_val /*[in]*/) {
  const char* env = std::getenv(name);
  if (env == nullptr) return default_val;

  std::vector<T> result;
  std::istringstream stream(env);
  std::string token;
  while (std::getline(stream, token, ',')) {
    result.push_back(ParseValue<T>(token));
  }
  return result;
}

//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
