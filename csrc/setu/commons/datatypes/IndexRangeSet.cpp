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
#include "commons/datatypes/IndexRangeSet.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
IndexRangeSet IndexRangeSet::MakeSingle(std::size_t dim_size,
                                        std::int64_t start, std::int64_t end) {
  ASSERT_VALID_ARGUMENTS(start >= 0, "Start {} must be non-negative", start);
  ASSERT_VALID_ARGUMENTS(start <= end, "Start {} must be <= end {}", start,
                         end);
  ASSERT_VALID_ARGUMENTS(static_cast<std::size_t>(end) <= dim_size,
                         "End {} must not exceed dim_size {}", end, dim_size);
  if (start == end) {
    return MakeEmpty(dim_size);
  }
  return IndexRangeSet(dim_size, {{start, end}});
}
//==============================================================================
IndexRangeSet IndexRangeSet::FromIndices(
    std::size_t dim_size, const std::set<std::int64_t>& indices) {
  if (indices.empty()) {
    return MakeEmpty(dim_size);
  }

  std::vector<IndexRange> result;
  auto it = indices.begin();
  std::int64_t range_start = *it;
  std::int64_t range_end = *it + 1;
  ++it;

  for (; it != indices.end(); ++it) {
    ASSERT_VALID_ARGUMENTS(*it >= 0 && static_cast<std::size_t>(*it) < dim_size,
                           "Index {} is out of bounds for dimension of size {}",
                           *it, dim_size);
    if (*it == range_end) {
      // Extend current range
      ++range_end;
    } else {
      // Flush current range and start new one
      result.push_back({range_start, range_end});
      range_start = *it;
      range_end = *it + 1;
    }
  }
  result.push_back({range_start, range_end});

  return IndexRangeSet(dim_size, std::move(result));
}
//==============================================================================
IndexRangeSet IndexRangeSet::Intersect(const IndexRangeSet& other) const {
  ASSERT_VALID_ARGUMENTS(dim_size == other.dim_size,
                         "Dimension sizes must match: {} vs {}", dim_size,
                         other.dim_size);

  if (ranges.empty() || other.ranges.empty()) {
    return MakeEmpty(dim_size);
  }

  std::vector<IndexRange> result;
  std::size_t i = 0;
  std::size_t j = 0;

  while (i < ranges.size() && j < other.ranges.size()) {
    std::int64_t lo = std::max(ranges[i].start, other.ranges[j].start);
    std::int64_t hi = std::min(ranges[i].end, other.ranges[j].end);

    if (lo < hi) {
      result.push_back({lo, hi});
    }

    // Advance the pointer with the smaller end
    if (ranges[i].end < other.ranges[j].end) {
      ++i;
    } else {
      ++j;
    }
  }

  return IndexRangeSet(dim_size, std::move(result));
}
//==============================================================================
IndexRangeSet IndexRangeSet::ShiftAndClamp(std::int64_t offset,
                                           std::size_t new_size) const {
  if (ranges.empty()) {
    return MakeEmpty(new_size);
  }

  const std::int64_t new_end = static_cast<std::int64_t>(new_size);
  std::vector<IndexRange> result;

  for (const auto& r : ranges) {
    std::int64_t lo = r.start - offset;
    std::int64_t hi = r.end - offset;

    // Clamp to [0, new_size)
    lo = std::max(lo, std::int64_t{0});
    hi = std::min(hi, new_end);

    if (lo < hi) {
      result.push_back({lo, hi});
    }
  }

  return IndexRangeSet(new_size, std::move(result));
}
//==============================================================================
void IndexRangeSet::Serialize(detail::BinaryBuffer& buffer) const {
  setu::commons::utils::BinaryWriter writer(buffer);
  writer.WriteFields(dim_size, ranges);
}
//==============================================================================
IndexRangeSet IndexRangeSet::Deserialize(const detail::BinaryRange& range) {
  setu::commons::utils::BinaryReader reader(range);
  auto [ds, r] = reader.ReadFields<std::size_t, std::vector<IndexRange>>();
  return IndexRangeSet(ds, std::move(r));
}
//==============================================================================
std::string IndexRangeSet::ToString() const {
  if (ranges.empty()) {
    return std::format("IndexRangeSet(dim_size={}, empty)", dim_size);
  }
  if (All()) {
    return std::format("IndexRangeSet(dim_size={}, full)", dim_size);
  }

  std::string result =
      std::format("IndexRangeSet(dim_size={}, ranges=[", dim_size);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    if (i > 0) result += ", ";
    result += std::format("[{}, {})", ranges[i].start, ranges[i].end);
  }
  result += "])";
  return result;
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
