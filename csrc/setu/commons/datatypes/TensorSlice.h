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
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Represents a contiguous slice of elements along a tensor dimension
 *
 * TensorSlice defines a logical selection of contiguous elements along a single
 * tensor dimension using start and end indices. This is fundamental for
 * defining how tensor data is partitioned and accessed in distributed
 * computations.
 */
struct TensorSlice {
  /**
   * @brief Constructs a tensor slice with start and end indices
   *
   * @param dim_name_param Name of the dimension being sliced
   * @param start_param Starting index (inclusive) of the slice
   * @param end_param Ending index (exclusive) of the slice
   *
   * @throws std::invalid_argument if start >= end
   */
  TensorSlice(TensorDimName dim_name_param, TensorIndex start_param,
              TensorIndex end_param)
      : dim_name(dim_name_param),
        start(start_param),
        end(end_param),
        size(end - start) {
    ASSERT_VALID_ARGUMENTS(start < end,
                           "Start index {} must be less than end index {}",
                           start, end);
  }

  /**
   * @brief Returns a string representation of the tensor slice
   *
   * @return String containing dimension name, start, end, and size
   */
  [[nodiscard]] std::string ToString() const {
    return std::format("TensorSlice(dim_name={}, start={}, end={}, size={})",
                       dim_name, start, end, size);
  }

  /**
   * @brief Converts the slice to a bitset representation
   *
   * Creates a bitset where bits corresponding to indices within the slice
   * are set to true, enabling efficient set operations for tensor selections.
   *
   * @param dim_size Total size of the dimension
   * @return Bitset with bits set for indices within the slice
   *
   * @throws std::invalid_argument if dim_size is smaller than slice size
   */
  [[nodiscard]] TensorIndicesBitset ToBitset(std::size_t dim_size) const {
    ASSERT_VALID_ARGUMENTS(
        dim_size >= size,
        "Dim size {} must be greater than or equal to slice size {}", dim_size,
        size);
    TensorIndicesBitset bitset(dim_size);
    bitset.set(start, size, true);
    return bitset;
  }

  const TensorDimName dim_name;  ///< Name of the dimension being sliced
  const TensorIndex start;       ///< Starting index (inclusive) of the slice
  const TensorIndex end;         ///< Ending index (exclusive) of the slice
  const std::size_t size;        ///< Size of the slice (end - start)
};
//==============================================================================
/// @brief Shared pointer to a TensorSlice object
using TensorSlicePtr = std::shared_ptr<TensorSlice>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
