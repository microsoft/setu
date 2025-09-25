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
#include "commons/utils/NumericalUtils.h"

namespace setu::commons::utils {

std::size_t RoundDownToNearestMultiple(std::size_t value,
                                       std::size_t multiple) {
  return (value / multiple) * multiple;
}

std::size_t RoundUpToNearestMultiple(std::size_t value, std::size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

}  // namespace setu::commons::utils
