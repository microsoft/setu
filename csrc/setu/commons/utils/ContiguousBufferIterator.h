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
#include "commons/datatypes/TensorSelection.h"
//==============================================================================
namespace setu::commons::utils {
using setu::commons::TensorDimName;
using setu::commons::TensorIndicesBitset;
using setu::commons::datatypes::TensorSelectionPtr;
//==============================================================================
struct ContiguousBufferRange {
  std::size_t
      start;  // The start offset from device_ptr, not including dtype size
  std::size_t length;  // Number of elements in the range

  [[nodiscard]] std::string ToString() const {
    return std::format("ContiguousBufferRange(start={}, length={})", start,
                       length);
  }
};
//==============================================================================
/**
 * @brief Computes and provides iteration over contiguous buffer ranges
 *
 * Given a TensorSelection and dimension ordering, this class computes the
 * contiguous memory ranges that correspond to the selected indices.
 *
 * The dimension names should be ordered from outermost to innermost. We assume
 * row-major encoding.
 *
 * The implementation eagerly computes all of the ranges. We will eventually
 * move to a lazy computation pattern.
 */
class ContiguousBufferRangeView {
 public:
  using Iterator = std::vector<ContiguousBufferRange>::const_iterator;

  ContiguousBufferRangeView(const std::vector<TensorDimName>& dim_names,
                            TensorSelectionPtr selection);

  [[nodiscard]] Iterator begin() const { return ranges_.begin(); }
  [[nodiscard]] Iterator end() const { return ranges_.end(); }

  [[nodiscard]] std::size_t size() const { return ranges_.size(); }
  [[nodiscard]] bool empty() const { return ranges_.empty(); }

 private:
  std::vector<ContiguousBufferRange> ranges_;

  static std::vector<ContiguousBufferRange> BitsetToRanges(
      const TensorIndicesBitset& bitset);

  void ComputeRanges(const std::vector<TensorDimName>& dim_names,
                     TensorSelectionPtr selection);
};
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
