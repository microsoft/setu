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
#include "commons/datatypes/TensorShardSpec.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================
void TensorShardSpec::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(name, dims, dtype, device);
}

TensorShardSpec TensorShardSpec::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [name_val, dims_val, dtype_val, device_val] =
      reader.ReadFields<TensorName, std::vector<TensorDimSpec>, torch::Dtype,
                        Device>();
  return TensorShardSpec(name_val, dims_val, dtype_val, device_val);
}
//==============================================================================
bool TensorShardSpec::Overlaps(const TensorShardSpec& other) const {
  // Shards overlap if and only if ALL dimensions overlap
  for (std::size_t i = 0; i < dims.size(); ++i) {
    // Ranges [start1, end1) and [start2, end2) overlap iff
    // start1 < end2 && start2 < end1
    bool dim_overlaps =
        dims[i].start < other.dims[i].end && other.dims[i].start < dims[i].end;
    if (!dim_overlaps) {
      return false;  // Found non-overlapping dimension
    }
  }
  return true;  // All dimensions overlap
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
