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
#include "commons/Types.h"
#include "commons/datatypes/TensorDim.h"
#include "commons/datatypes/TensorDimShard.h"
#include "commons/datatypes/TensorShard.h"
#include "commons/datatypes/TensorSlice.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::datatypes {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
//==============================================================================
// Forward declarations
struct TensorSelection;
using TensorSelectionPtr = std::shared_ptr<TensorSelection>;
//==============================================================================
struct TensorSelection {
  TensorSelection(TensorName name_param, TensorDimMap dims_param)
      : name(name_param), indices(BuildIndicesFromDims(dims_param)) {
    ASSERT_VALID_ARGUMENTS(dims_param.size() > 0, "Dims must be non-empty");
  }

  TensorSelection(TensorName name_param, TensorIndicesMap indices_param)
      : name(name_param), indices(indices_param) {
    ASSERT_VALID_ARGUMENTS(indices_param.size() > 0,
                           "Indices must be non-empty");
  }

  [[nodiscard]] std::string ToString() const {
    return std::format("TensorSelection(name={}, indices={})", name, indices);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static TensorSelection Deserialize(const BinaryRange& range);

  [[nodiscard]] TensorSelectionPtr GetIntersection(
      TensorSelectionPtr other) const {
    ASSERT_VALID_POINTER_ARGUMENT(other);
    ASSERT_VALID_ARGUMENTS(name == other->name, "Selection names do not match");
    ASSERT_VALID_ARGUMENTS(
        indices.size() == other->indices.size(),
        "Selections have different number of dimensions: {} vs {}",
        indices.size(), other->indices.size());

    TensorIndicesMap intersection;
    for (const auto& [dim_name, dim] : indices) {
      ASSERT_VALID_ARGUMENTS(
          other->indices.find(dim_name) != other->indices.end(),
          "Dim {} not found in other selection", dim_name);

      intersection[dim_name] =
          indices.at(dim_name).Intersect(other->indices.at(dim_name));
    }
    return std::make_shared<TensorSelection>(name, intersection);
  }

  [[nodiscard]] bool IsSpanning() const {
    for (const auto& [dim_name, dim] : indices) {
      if (!dim.All()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool IsEmpty() const {
    for (const auto& [dim_name, dim] : indices) {
      if (dim.None()) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool IsCompatible(TensorSelectionPtr other) const {
    // First we need to check if the dimensions are the same
    if (indices.size() != other->indices.size()) {
      return false;
    }

    // Now we need to make sure that the size of the dimensions are the same
    for (const auto& [dim_name, dim] : indices) {
      if (dim.Size() != other->indices.at(dim_name).Size()) {
        return false;
      }
    }

    return true;
  }

  /**
   * @brief Check equality with another TensorSelection
   *
   * Two TensorSelections are equal if they have the same name and identical
   * index bitsets for all dimensions.
   *
   * @param other The TensorSelection to compare against
   * @return true if both selections are identical, false otherwise
   */
  [[nodiscard]] bool operator==(const TensorSelection& other) const {
    if (name != other.name) {
      return false;
    }
    if (indices.size() != other.indices.size()) {
      return false;
    }
    for (const auto& [dim_name, dim] : indices) {
      auto it = other.indices.find(dim_name);
      if (it == other.indices.end()) {
        return false;
      }
      if (dim != it->second) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool operator!=(const TensorSelection& other) const {
    return !(*this == other);
  }

  /**
   * @brief Get the indices bitset for a specific dimension
   *
   * @param dim_name The name of the dimension
   * @return const reference to the TensorIndices for the dimension
   * @throws std::invalid_argument if dimension is not found
   */
  [[nodiscard]] const TensorIndices& GetDimIndices(
      const TensorDimName& dim_name) const {
    auto it = indices.find(dim_name);
    ASSERT_VALID_ARGUMENTS(it != indices.end(),
                           "Dimension {} not found in selection", dim_name);
    return it->second;
  }

  /**
   * @brief Create a new TensorSelection with specified indices for a dimension
   *
   * @param dim_name The name of the dimension to select from
   * @param index_set The set of indices to select
   * @return New TensorSelection with the specified indices for the dimension
   */
  [[nodiscard]] TensorSelectionPtr Where(
      const TensorDimName& dim_name,
      const std::set<TensorIndex>& index_set) const {
    ASSERT_VALID_ARGUMENTS(indices.find(dim_name) != indices.end(),
                           "Dimension {} not found in selection", dim_name);

    TensorIndicesMap new_indices = indices;

    const auto& current = indices.at(dim_name);
    auto from_set = TensorIndices::FromIndices(current.Size(), index_set);

    // Intersect with current selection (only keep indices that are both
    // selected and requested)
    new_indices[dim_name] = current.Intersect(from_set);

    return std::make_shared<TensorSelection>(name, new_indices);
  }

  /**
   * @brief Create a new TensorSelection with specified slice for a dimension
   *
   * @param dim_name The name of the dimension to select from
   * @param slice The tensor slice to apply
   * @return New TensorSelection with the specified slice for the dimension
   */
  [[nodiscard]] TensorSelectionPtr Where(const TensorDimName& dim_name,
                                         TensorSlicePtr slice) const {
    ASSERT_VALID_POINTER_ARGUMENT(slice);
    ASSERT_VALID_ARGUMENTS(indices.find(dim_name) != indices.end(),
                           "Dimension {} not found in selection", dim_name);

    // Create a copy of current indices
    TensorIndicesMap new_indices = indices;

    const auto& current = indices.at(dim_name);
    auto slice_range =
        TensorIndices::MakeSingle(current.Size(), slice->start, slice->end);

    // Intersect with current selection (only keep indices that are both
    // selected and in slice)
    new_indices[dim_name] = current.Intersect(slice_range);

    return std::make_shared<TensorSelection>(name, new_indices);
  }

  /**
   * @brief Localize this selection to a shard's coordinate space
   *
   * Creates a new TensorSelection where the bitsets are sized to the shard's
   * owned region and indices are shifted to be relative to the shard's start.
   *
   * Example: If dimension "x" has size 100 in the full tensor, the shard owns
   * [25, 50), and this selection has indices {30, 31, 35} selected, the
   * localized selection will have indices {5, 6, 10} in a bitset of size 25.
   *
   * This is useful for getting shard-local buffer offsets from
   * ContiguousBufferRangeView.
   *
   * @param shard The shard metadata defining the local coordinate space
   * @return New TensorSelection in the shard's local coordinate space
   */
  [[nodiscard]] TensorSelectionPtr Localize(
      TensorShardMetadataPtr shard) const {
    ASSERT_VALID_POINTER_ARGUMENT(shard);
    ASSERT_VALID_ARGUMENTS(
        name == shard->spec.name,
        "Selection tensor name {} does not match shard tensor name {}", name,
        shard->spec.name);

    TensorIndicesMap localized_indices;
    for (const auto& dim_spec : shard->spec.dims) {
      std::size_t local_size = dim_spec.GetOwnedSize();
      const auto& current = indices.at(dim_spec.name);
      localized_indices[dim_spec.name] =
          current.ShiftAndClamp(dim_spec.start, local_size);
    }

    return std::make_shared<TensorSelection>(name, localized_indices);
  }

  const TensorName name;

 private:
  static TensorIndicesMap BuildIndicesFromDims(TensorDimMap dims_param) {
    TensorIndicesMap result_indices;
    for (const auto& [dim_name, dim] : dims_param) {
      result_indices[dim_name] = TensorIndices::MakeFull(dim.size);
    }
    return result_indices;
  }

  const TensorIndicesMap indices;
};
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
    result_indices[dim_spec.name] =
        TensorIndices::MakeSingle(dim_spec.size, dim_spec.start, dim_spec.end);
  }
  return std::make_shared<TensorSelection>(spec->name, result_indices);
}
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
  return CreateSelectionFromShardSpec(
      std::make_shared<TensorShardSpec>(shard->metadata.spec));
}
//==============================================================================
/**
 * @brief Create a TensorSelection from a TensorShardMetadata
 *
 * This utility function creates a TensorSelection that represents the exact
 * region of the tensor owned by the given shard.
 *
 * @param shard The TensorShardMetadata to create a selection from
 * @return TensorSelectionPtr A selection covering the shard's region
 */
inline TensorSelectionPtr CreateSelectionFromShardMetadata(
    TensorShardMetadataPtr shard_metadata) {
  ASSERT_VALID_POINTER_ARGUMENT(shard_metadata);
  return CreateSelectionFromShardSpec(
      std::make_shared<TensorShardSpec>(shard_metadata->spec));
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================