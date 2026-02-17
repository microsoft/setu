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
#include "planner/ir/llc/ShardRef.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

std::string ShardRef::ToString() const {
  std::string name_str =
      tensor_name.has_value() ? tensor_name.value() : "<none>";
  std::string node_str =
      node_id.has_value() ? boost::uuids::to_string(node_id.value()) : "<none>";
  return std::format("ShardRef(shard_id={}, tensor_name={}, node_id={})",
                     boost::uuids::to_string(shard_id), name_str, node_str);
}

void ShardRef::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(shard_id, node_id, tensor_name);
}

ShardRef ShardRef::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [shard_id, node_id, tensor_name] =
      reader.ReadFields<ShardId, std::optional<NodeId>,
                        std::optional<TensorName>>();

  ShardRef ref(std::move(shard_id), std::move(tensor_name));
  ref.node_id = std::move(node_id);
  return ref;
}

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
