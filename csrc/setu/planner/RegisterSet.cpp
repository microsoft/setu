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
#include "planner/RegisterSet.h"
//==============================================================================
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::planner {
//==============================================================================
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

void RegisterSet::Serialize(setu::commons::BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.Write(sizes_);
}

RegisterSet RegisterSet::Deserialize(const setu::commons::BinaryRange& range) {
  BinaryReader reader(range);
  RegisterSet set;
  set.sizes_ = reader.Read<std::vector<std::size_t>>();
  return set;
}

//==============================================================================
}  // namespace setu::planner
//==============================================================================
