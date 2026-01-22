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
#include "commons/datatypes/TensorSelection.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================
void TensorSelection::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.Write(name);
  writer.Write<std::uint32_t>(static_cast<std::uint32_t>(indices.size()));
  for (const auto& [dim_name, dim_indices] : indices) {
    writer.Write(dim_name);
    SerializeBitset(writer, dim_indices);
  }
}

TensorSelection TensorSelection::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto name_val = reader.Read<TensorName>();
  const auto indices_size = reader.Read<std::uint32_t>();
  TensorIndicesMap indices_val;
  indices_val.reserve(indices_size);
  for (std::uint32_t i = 0; i < indices_size; ++i) {
    auto dim_name = reader.Read<TensorDimName>();
    auto bitset = DeserializeBitset(reader);
    indices_val.emplace(dim_name, std::move(bitset));
  }
  return TensorSelection(name_val, indices_val);
}

void TensorSelection::SerializeBitset(BinaryWriter& writer,
                                      const TensorIndicesBitset& bitset) {
  writer.Write<std::uint64_t>(static_cast<std::uint64_t>(bitset.size()));
  std::vector<TensorIndicesBitset::block_type> blocks(bitset.num_blocks());
  boost::to_block_range(bitset, blocks.begin());
  writer.Write(blocks);
}

TensorIndicesBitset TensorSelection::DeserializeBitset(BinaryReader& reader) {
  const auto bit_count = static_cast<std::size_t>(reader.Read<std::uint64_t>());
  auto blocks = reader.Read<std::vector<TensorIndicesBitset::block_type>>();
  TensorIndicesBitset bitset(bit_count);
  boost::from_block_range(blocks.begin(), blocks.end(), bitset);
  return bitset;
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
