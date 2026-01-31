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
#include "commons/utils/ContiguousBufferIterator.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
ContiguousBufferRangeView::ContiguousBufferRangeView(
    const std::vector<TensorDimName>& dim_names, TensorSelectionPtr selection) {
  ASSERT_VALID_POINTER_ARGUMENT(selection);
  ComputeRanges(dim_names, selection);
}
//==============================================================================
std::vector<ContiguousBufferRange> ContiguousBufferRangeView::BitsetToRanges(
    const TensorIndicesBitset& bitset) {
  std::vector<ContiguousBufferRange> ranges;

  auto pos = bitset.find_first();
  if (pos == TensorIndicesBitset::npos) {
    return ranges;
  }

  std::size_t start = pos;
  std::size_t length = 1;

  while ((pos = bitset.find_next(pos)) != TensorIndicesBitset::npos) {
    if (pos == start + length) {
      ++length;
    } else {
      ranges.push_back({start, length});
      start = pos;
      length = 1;
    }
  }
  ranges.push_back({start, length});

  return ranges;
}
//==============================================================================
void ContiguousBufferRangeView::ComputeRanges(
    const std::vector<TensorDimName>& dim_names, TensorSelectionPtr selection) {
  if (dim_names.empty()) {
    return;
  }

  // Check for empty selections
  for (const auto& name : dim_names) {
    if (selection->GetDimIndices(name).none()) {
      return;
    }
  }

  // Start with innermost dimension (last in the vector)
  const auto& innermost = selection->GetDimIndices(dim_names.back());
  ranges_ = BitsetToRanges(innermost);
  std::size_t block_size = innermost.size();

  // Process dimensions from inner to outer (reverse iteration excluding last)
  for (std::size_t i = dim_names.size() - 1; i-- > 0;) {
    const auto& bitset = selection->GetDimIndices(dim_names[i]);
    const std::size_t stride = block_size;

    const bool is_full_range = (ranges_.size() == 1 && ranges_[0].start == 0 &&
                                ranges_[0].length == stride);

    if (is_full_range) {
      // Optimization: if inner dimensions are fully selected, just scale
      ranges_ = BitsetToRanges(bitset);
      for (auto& range : ranges_) {
        range.start *= stride;
        range.length *= stride;
      }
    } else {
      // General case: replicate ranges for each selected index
      std::vector<ContiguousBufferRange> new_ranges;

      for (auto idx = bitset.find_first(); idx != TensorIndicesBitset::npos;
           idx = bitset.find_next(idx)) {
        const std::size_t base = idx * stride;

        for (const auto& range : ranges_) {
          const std::size_t new_start = base + range.start;

          // Merge with previous range if contiguous
          if (!new_ranges.empty() &&
              new_ranges.back().start + new_ranges.back().length == new_start) {
            new_ranges.back().length += range.length;
          } else {
            new_ranges.push_back({new_start, range.length});
          }
        }
      }
      ranges_ = std::move(new_ranges);
    }

    block_size *= bitset.size();
  }
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
