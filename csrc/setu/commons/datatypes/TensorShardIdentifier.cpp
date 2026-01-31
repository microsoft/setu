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
#include "commons/datatypes/TensorShardIdentifier.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

std::string TensorShardIdentifier::ToString() const {
  return std::format("TensorShardIdentifier(name={}, shard_id={})", tensor_name,
                     boost::uuids::to_string(shard_id));
}

void TensorShardIdentifier::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(tensor_name, shard_id);
}

TensorShardIdentifier TensorShardIdentifier::Deserialize(
    const BinaryRange& range) {
  BinaryReader reader(range);
  auto [name, id] = reader.ReadFields<TensorName, ShardId>();
  return TensorShardIdentifier(std::move(name), std::move(id));
}

//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
