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
/**
 * Rounds down a value to the nearest multiple of the specified value.
 *
 * @param value The value to round down.
 * @param multiple The multiple to round down to.
 * @return The value rounded down to the nearest multiple.
 */
std::size_t RoundDownToNearestMultiple(std::size_t value, std::size_t multiple);
//==============================================================================
/**
 * Rounds up a value to the nearest multiple of the specified value.
 *
 * @param value The value to round up.
 * @param multiple The multiple to round up to.
 * @return The value rounded up to the nearest multiple.
 */
std::size_t RoundUpToNearestMultiple(std::size_t value, std::size_t multiple);
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
