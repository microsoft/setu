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
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/TensorShard.h"
#include "commons/datatypes/TensorShardSpec.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::TensorIndex;
using setu::commons::TensorIndicesBitset;
using setu::commons::TensorIndicesMap;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorShardPtr;
using setu::commons::datatypes::TensorShardSpecPtr;
//==============================================================================
// Forward declaration to avoid circular dependency
namespace setu::commons::datatypes {
struct TensorSelection;
using TensorSelectionPtr = std::shared_ptr<TensorSelection>;
}  // namespace setu::commons::datatypes
using setu::commons::datatypes::TensorSelection;
using setu::commons::datatypes::TensorSelectionPtr;
//==============================================================================
/**
 * @brief Create a TensorSelection from a TensorShard
 *
 * This utility function creates a TensorSelection that represents the exact
 * region of the tensor owned by the given shard.
 *
 * @param shard The TensorShard to create a selection from
 * @return TensorSelectionPtr A selection covering the shard's region
 */
inline TensorSelectionPtr CreateSelectionFromShard(TensorShardPtr shard) {
  ASSERT_VALID_POINTER_ARGUMENT(shard);

  TensorIndicesMap result_indices;
  for (const auto& [dim_name, dim_shard] : shard->dim_shards) {
    // Create bitset for the full dimension size
    TensorIndicesBitset bitset(dim_shard.dim_size);
    // Set bits only for the slice owned by this shard
    for (TensorIndex i = dim_shard.slice->start; i < dim_shard.slice->end;
         ++i) {
      bitset[static_cast<std::size_t>(i)] = true;
    }
    result_indices[dim_name] = bitset;
  }
  return std::make_shared<TensorSelection>(shard->name, result_indices);
}
//==============================================================================
/**
 * @brief Create a TensorSelection from a TensorShardSpec
 *
 * This utility function creates a TensorSelection that represents the exact
 * region of the tensor defined by the given shard specification. Unlike
 * CreateSelectionFromShard, this works with TensorShardSpec which uses a vector
 * of TensorDimSpec instead of a map of TensorDimShard.
 *
 * @param spec The TensorShardSpec to create a selection from
 * @return TensorSelectionPtr A selection covering the spec's region
 */
inline TensorSelectionPtr CreateSelectionFromShardSpec(
    TensorShardSpecPtr spec) {
  ASSERT_VALID_POINTER_ARGUMENT(spec);

  TensorIndicesMap result_indices;
  for (const auto& dim_spec : spec->dims) {
    // Create bitset for the full dimension size
    TensorIndicesBitset bitset(dim_spec.size);
    // Set bits only for the range owned by this shard [start, end)
    for (TensorIndex i = dim_spec.start; i < dim_spec.end; ++i) {
      bitset[static_cast<std::size_t>(i)] = true;
    }
    result_indices[dim_spec.name] = bitset;
  }
  return std::make_shared<TensorSelection>(spec->name, result_indices);
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
