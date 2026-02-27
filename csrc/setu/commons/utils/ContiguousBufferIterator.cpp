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
/**
 * @brief Convert IndexRangeSet ranges to ContiguousBufferRange vector
 *
 * Simple conversion from the IndexRangeSet's ranges to our buffer range format.
 */
static std::vector<ContiguousBufferRange> ToBufferRanges(
    const TensorIndices& index_set) {
  std::vector<ContiguousBufferRange> result;
  result.reserve(index_set.ranges.size());
  for (const auto& r : index_set.ranges) {
    result.push_back({static_cast<std::size_t>(r.start),
                      static_cast<std::size_t>(r.end - r.start)});
  }
  return result;
}
//==============================================================================
void ContiguousBufferRangeView::ComputeRanges(
    const std::vector<TensorDimName>& dim_names, TensorSelectionPtr selection) {
  if (dim_names.empty()) {
    return;
  }

  // Check for empty selections
  for (const auto& name : dim_names) {
    if (selection->GetDimIndices(name).None()) {
      return;
    }
  }

  // Start with innermost dimension (last in the vector)
  const auto& innermost = selection->GetDimIndices(dim_names.back());
  ranges_ = ToBufferRanges(innermost);
  std::size_t block_size = innermost.Size();

  // Process dimensions from inner to outer (reverse iteration excluding last)
  for (std::size_t i = dim_names.size() - 1; i-- > 0;) {
    const auto& dim_index_set = selection->GetDimIndices(dim_names[i]);
    const std::size_t stride = block_size;

    const bool is_full_range = (ranges_.size() == 1 && ranges_[0].start == 0 &&
                                ranges_[0].length == stride);

    if (is_full_range) {
      // Optimization: if inner dimensions are fully selected, just scale
      ranges_ = ToBufferRanges(dim_index_set);
      for (auto& range : ranges_) {
        range.start *= stride;
        range.length *= stride;
      }
    } else {
      // General case: replicate ranges for each selected index in this dim
      std::vector<ContiguousBufferRange> new_ranges;

      for (const auto& outer_range : dim_index_set.ranges) {
        for (std::int64_t idx = outer_range.start; idx < outer_range.end;
             ++idx) {
          const std::size_t base = static_cast<std::size_t>(idx) * stride;

          for (const auto& range : ranges_) {
            const std::size_t new_start = base + range.start;

            // Merge with previous range if contiguous
            if (!new_ranges.empty() &&
                new_ranges.back().start + new_ranges.back().length ==
                    new_start) {
              new_ranges.back().length += range.length;
            } else {
              new_ranges.push_back({new_start, range.length});
            }
          }
        }
      }
      ranges_ = std::move(new_ranges);
    }

    block_size *= dim_index_set.Size();
  }
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
