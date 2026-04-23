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
//==============================================================================
namespace setu::planner::targets {
//==============================================================================

/// Stores (interval, value) pairs over a std::size_t offset space.
///
/// No invariant about disjointness of stored intervals: callers that want
/// "latest writer per byte" semantics pair Insert with SupersedeRange;
/// callers that want "all readers since the last write" semantics just
/// Insert and periodically SupersedeRange to clear.
///
/// The public interface intentionally hides the representation so a
/// different backing store (for example a balanced interval tree with
/// max-end augmentation) can replace the flat-vector body without
/// touching call sites.
template <typename V>
class IntervalMap {
 public:
  struct Entry {
    std::size_t start;  ///< inclusive byte offset
    std::size_t end;    ///< exclusive byte offset
    V value;
  };

  class OverlapIterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const Entry;
    using reference = const Entry&;
    using pointer = const Entry*;
    using difference_type = std::ptrdiff_t;

    OverlapIterator() = default;

    [[nodiscard]] reference operator*() const { return *it_; }
    [[nodiscard]] pointer operator->() const { return &*it_; }

    OverlapIterator& operator++() {
      ++it_;
      AdvanceToOverlap();
      return *this;
    }

    OverlapIterator operator++(int) {
      auto copy = *this;
      ++*this;
      return copy;
    }

    [[nodiscard]] bool operator==(const OverlapIterator& other) const {
      return it_ == other.it_;
    }
    [[nodiscard]] bool operator!=(const OverlapIterator& other) const {
      return !(*this == other);
    }

   private:
    friend class IntervalMap;
    using Inner = typename std::vector<Entry>::const_iterator;

    OverlapIterator(Inner it, Inner end, std::size_t query_start,
                    std::size_t query_end)
        : it_(it),
          end_(end),
          query_start_(query_start),
          query_end_(query_end) {
      AdvanceToOverlap();
    }

    void AdvanceToOverlap() {
      while (it_ != end_ && !(it_->start < query_end_ && query_start_ < it_->end)) {
        ++it_;
      }
    }

    Inner it_{};
    Inner end_{};
    std::size_t query_start_ = 0;
    std::size_t query_end_ = 0;
  };

  class OverlapRange {
   public:
    [[nodiscard]] OverlapIterator begin() const { return begin_; }
    [[nodiscard]] OverlapIterator end() const { return end_; }

   private:
    friend class IntervalMap;
    OverlapRange(OverlapIterator begin_param, OverlapIterator end_param)
        : begin_(begin_param), end_(end_param) {}

    OverlapIterator begin_;
    OverlapIterator end_;
  };

  /// Returns a lazy range over every entry whose [start, end) overlaps
  /// [query_start, query_end).  Iteration order is unspecified.
  [[nodiscard]] OverlapRange Overlaps(std::size_t query_start,
                                      std::size_t query_end) const {
    return OverlapRange(
        OverlapIterator(entries_.begin(), entries_.end(), query_start,
                        query_end),
        OverlapIterator(entries_.end(), entries_.end(), query_start,
                        query_end));
  }

  /// Appends an entry.  Does not check for overlap with existing entries.
  void Insert(std::size_t start, std::size_t end, V value) {
    ASSERT_VALID_ARGUMENTS(start < end,
                           "IntervalMap::Insert: empty interval [{}, {})",
                           start, end);
    entries_.push_back(Entry{start, end, std::move(value)});
  }

  /// Trims or erases every entry so that no byte in [range_start, range_end)
  /// remains covered.  Entries fully covered are erased; entries straddling
  /// one boundary are trimmed; entries straddling both boundaries split
  /// into two remainders.
  void SupersedeRange(std::size_t range_start, std::size_t range_end) {
    ASSERT_VALID_ARGUMENTS(
        range_start < range_end,
        "IntervalMap::SupersedeRange: empty range [{}, {})", range_start,
        range_end);

    // pop-and-swap every overlapping entry; push surviving remnants
    // (up to two per original entry).
    std::size_t i = 0;
    while (i < entries_.size()) {
      const auto& current = entries_[i];
      const bool overlaps =
          current.start < range_end && range_start < current.end;
      if (!overlaps) {
        ++i;
        continue;
      }

      const auto saved_start = current.start;
      const auto saved_end = current.end;
      V saved_value = std::move(entries_[i].value);

      entries_[i] = std::move(entries_.back());
      entries_.pop_back();
      // Do NOT increment i; the swapped-in element still needs to be
      // checked.

      if (saved_start < range_start) {
        entries_.push_back(Entry{saved_start, range_start, saved_value});
      }
      if (range_end < saved_end) {
        entries_.push_back(
            Entry{range_end, saved_end, std::move(saved_value)});
      }
    }
  }

  void Clear() { entries_.clear(); }
  [[nodiscard]] bool Empty() const { return entries_.empty(); }
  [[nodiscard]] std::size_t Size() const { return entries_.size(); }

 private:
  std::vector<Entry> entries_;
};

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
