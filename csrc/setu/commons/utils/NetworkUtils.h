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
// Get the local IP address
[[nodiscard]] std::string GetLocalIpAddress();
//==============================================================================
// Generate a random port that is not in use
[[nodiscard]] std::size_t GetRandomPort();
//==============================================================================
// Generate multiple random ports
[[nodiscard]] std::vector<std::size_t> GetRandomPorts(int n);
//==============================================================================
// Check if a port is in use
[[nodiscard]] bool IsPortInUse(std::size_t port);
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
