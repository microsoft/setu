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
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
// Forward declarations for serialization types
namespace detail {
using BinaryBuffer = std::vector<std::uint8_t>;
using BinaryIterator = BinaryBuffer::const_iterator;
using BinaryRange = std::pair<BinaryIterator, BinaryIterator>;
}  // namespace detail
//==============================================================================
/**
 * @brief A half-open range [start, end) of indices
 *
 * Trivially copyable struct for efficient storage and serialization.
 */
struct IndexRange {
  std::int64_t start;  ///< Inclusive start index
  std::int64_t end;    ///< Exclusive end index

  [[nodiscard]] std::int64_t Length() const { return end - start; }

  [[nodiscard]] bool operator==(const IndexRange& other) const {
    return start == other.start && end == other.end;
  }
};
//==============================================================================
/**
 * @brief Represents a set of indices as sorted, non-overlapping ranges
 *
 * Replaces boost::dynamic_bitset<> for representing tensor dimension
 * selections. For the common case of a single contiguous range per dimension,
 * this uses 24 bytes instead of 8 MB (for a 64M-element dimension).
 *
 * All operations are O(R) where R is the number of ranges (typically 1),
 * compared to O(dim_size) for the bitset representation.
 */
struct IndexRangeSet {
  std::size_t dim_size;            ///< Total dimension size
  std::vector<IndexRange> ranges;  ///< Sorted, non-overlapping ranges

  // ---- Constructors ----

  IndexRangeSet() : dim_size(0) {}

  IndexRangeSet(std::size_t dim_size_param,
                std::vector<IndexRange> ranges_param)
      : dim_size(dim_size_param), ranges(std::move(ranges_param)) {}

  // ---- Factories ----

  /** @brief Create a set selecting all indices [0, dim_size) */
  [[nodiscard]] static IndexRangeSet MakeFull(std::size_t dim_size) {
    if (dim_size == 0) {
      return IndexRangeSet(0, {});
    }
    return IndexRangeSet(dim_size, {{0, static_cast<std::int64_t>(dim_size)}});
  }

  /** @brief Create an empty set (no indices selected) */
  [[nodiscard]] static IndexRangeSet MakeEmpty(std::size_t dim_size) {
    return IndexRangeSet(dim_size, {});
  }

  /** @brief Create a set with a single range [start, end) */
  [[nodiscard]] static IndexRangeSet MakeSingle(std::size_t dim_size,
                                                std::int64_t start,
                                                std::int64_t end);

  /** @brief Create from a set of arbitrary indices, coalescing consecutive
   * ones into ranges */
  [[nodiscard]] static IndexRangeSet FromIndices(
      std::size_t dim_size, const std::set<std::int64_t>& indices);

  // ---- Predicates ----

  /** @brief Check if all indices are selected */
  [[nodiscard]] bool All() const {
    return ranges.size() == 1 && ranges[0].start == 0 &&
           ranges[0].end == static_cast<std::int64_t>(dim_size);
  }

  /** @brief Check if no indices are selected */
  [[nodiscard]] bool None() const { return ranges.empty(); }

  /** @brief Return the dimension size (analogous to bitset.size()) */
  [[nodiscard]] std::size_t Size() const { return dim_size; }

  /** @brief Return the total number of selected elements */
  [[nodiscard]] std::size_t Count() const {
    std::size_t total = 0;
    for (const auto& r : ranges) {
      total += static_cast<std::size_t>(r.end - r.start);
    }
    return total;
  }

  // ---- Operations ----

  /** @brief Compute intersection with another IndexRangeSet (O(R1+R2)) */
  [[nodiscard]] IndexRangeSet Intersect(const IndexRangeSet& other) const;

  /**
   * @brief Shift ranges by -offset and clamp to [0, new_size)
   *
   * This replaces the bitset >> shift + resize pattern used in Localize.
   * Each range [a, b) becomes [a - offset, b - offset) clamped to [0,
   * new_size).
   */
  [[nodiscard]] IndexRangeSet ShiftAndClamp(std::int64_t offset,
                                            std::size_t new_size) const;

  /** @brief Equality comparison */
  [[nodiscard]] bool operator==(const IndexRangeSet& other) const {
    return dim_size == other.dim_size && ranges == other.ranges;
  }

  [[nodiscard]] bool operator!=(const IndexRangeSet& other) const {
    return !(*this == other);
  }

  // ---- Serialization ----

  void Serialize(detail::BinaryBuffer& buffer) const;

  static IndexRangeSet Deserialize(const detail::BinaryRange& range);

  // ---- Display ----

  [[nodiscard]] std::string ToString() const;
};
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
