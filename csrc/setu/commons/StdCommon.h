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
// C headers
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
//==============================================================================
// C++ headers
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
// Note: <format> requires C++20 and may not be available in all environments
// Using alternative string formatting approaches where needed
// #include <format>  // NOLINT
#include <fstream>
#include <functional>
#include <future>  // NOLINT(build/c++11)
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <numeric>
#include <optional>
#include <random>
#include <ranges>  // NOLINT(build/include_order)
#include <set>
#include <span>  // NOLINT
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
//==============================================================================
namespace std {
/// @brief Hash specialization for std::set<int>
///
/// Provides custom hash function for std::set<int> to enable usage as key in
/// unordered containers.
template <>
struct hash<std::set<int>> {
  std::size_t operator()(const std::set<int>& s) const {
    std::size_t hash = 0;
    for (int x : s) {
      hash ^= std::hash<int>{}(x) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};
}  // namespace std
//==============================================================================
