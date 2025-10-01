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
//==============================================================================
namespace setu::commons::datatypes {
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
          indices.at(dim_name) & other->indices.at(dim_name);
    }
    return std::make_shared<TensorSelection>(name, intersection);
  }

  [[nodiscard]] bool IsSpanning() const {
    for (const auto& [dim_name, dim] : indices) {
      if (!dim.all()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool IsEmpty() const {
    for (const auto& [dim_name, dim] : indices) {
      if (!dim.none()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool IsCompatible(TensorSelectionPtr other) const {
    // First we need to check if the dimensions are the same
    if (indices.size() != other->indices.size()) {
      return false;
    }

    // Now we need to make sure that the size of the dimensions are the same
    for (const auto& [dim_name, dim] : indices) {
      if (dim.size() != other->indices.at(dim_name).size()) {
        return false;
      }
    }

    return true;
  }

  /**
   * @brief Create a new TensorSelection with specified indices for a dimension
   *
   * @param dim_name The name of the dimension to select from
   * @param index_set The set of indices to select
   * @return New TensorSelection with the specified indices for the dimension
   */
  [[nodiscard]] TensorSelectionPtr Where(
      const TensorDimName& dim_name, const TensorIndicesPtr index_set) const {
    ASSERT_VALID_ARGUMENTS(indices.find(dim_name) != indices.end(),
                           "Dimension {} not found in selection", dim_name);

    // Create a copy of current indices
    TensorIndicesMap new_indices = indices;

    // Convert the index set to a bitset
    const auto& current_bitset = indices.at(dim_name);
    TensorIndicesBitset new_bitset(current_bitset.size());

    for (TensorIndex idx : *index_set) {
      ASSERT_VALID_ARGUMENTS(
          idx >= 0 && static_cast<std::size_t>(idx) < current_bitset.size(),
          "Index {} is out of bounds for dimension {} (size: {})", idx,
          dim_name, current_bitset.size());
      new_bitset[static_cast<std::size_t>(idx)] = true;
    }

    // Intersect with current selection (only keep indices that are both
    // selected and requested)
    new_indices[dim_name] = current_bitset & new_bitset;

    return std::make_shared<TensorSelection>(name, new_indices);
  }

  [[nodiscard]] TensorSelectionPtr Where(
      const TensorDimName& dim_name,
      const std::set<TensorIndex>& index_set) const {
    return Where(dim_name, std::make_shared<std::set<TensorIndex>>(index_set));
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

    // Convert the slice to a bitset
    const auto& current_bitset = indices.at(dim_name);
    TensorIndicesBitset slice_bitset = slice->ToBitset(current_bitset.size());

    // Intersect with current selection (only keep indices that are both
    // selected and in slice)
    new_indices[dim_name] = current_bitset & slice_bitset;

    return std::make_shared<TensorSelection>(name, new_indices);
  }

  const TensorName name;

 private:
  static TensorIndicesMap BuildIndicesFromDims(TensorDimMap dims_param) {
    TensorIndicesMap result_indices;
    for (const auto& [dim_name, dim] : dims_param) {
      // Initialize with all bits set (selecting all indices by default)
      TensorIndicesBitset bitset(dim.size);
      bitset.set();  // Set all bits to 1
      result_indices[dim_name] = bitset;
    }
    return result_indices;
  }

  const TensorIndicesMap indices;
};
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
