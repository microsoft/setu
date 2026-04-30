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
/// Backed by an AVL-balanced interval tree with subtree-max-end
/// augmentation. Overlap queries are O(log N + k) where k is the number
/// of entries that actually overlap the query, not the number of entries
/// with `start < query_end`. This is the real fix for workloads with many
/// scattered intervals where a flat-vector linear scan or a non-augmented
/// std::multimap both degrade to O(N) per query.
///
/// No invariant about disjointness of stored intervals: callers that want
/// "latest writer per byte" semantics pair Insert with SupersedeRange;
/// callers that want "all readers since the last write" semantics just
/// Insert and periodically SupersedeRange to clear.
template <typename V>
class IntervalMap {
 public:
  struct Entry {
    std::size_t start;  ///< inclusive byte offset
    std::size_t end;    ///< exclusive byte offset
    V value;
  };

 private:
  struct Node {
    std::size_t start;
    std::size_t end;
    V value;
    std::size_t subtree_max_end;  ///< max(end) over self and descendants
    int height;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    Node(std::size_t s, std::size_t e, V v)
        : start(s), end(e), value(std::move(v)), subtree_max_end(e),
          height(1) {}
  };

  static int Height(const std::unique_ptr<Node>& n) {
    return n ? n->height : 0;
  }
  static std::size_t MaxEnd(const std::unique_ptr<Node>& n) {
    return n ? n->subtree_max_end : 0;
  }
  static void Refresh(std::unique_ptr<Node>& n) {
    if (!n) return;
    n->height = 1 + std::max(Height(n->left), Height(n->right));
    n->subtree_max_end =
        std::max({n->end, MaxEnd(n->left), MaxEnd(n->right)});
  }
  static int Balance(const std::unique_ptr<Node>& n) {
    return n ? Height(n->left) - Height(n->right) : 0;
  }
  static std::unique_ptr<Node> RotateRight(std::unique_ptr<Node> y) {
    auto x = std::move(y->left);
    y->left = std::move(x->right);
    Refresh(y);
    x->right = std::move(y);
    Refresh(x);
    return x;
  }
  static std::unique_ptr<Node> RotateLeft(std::unique_ptr<Node> x) {
    auto y = std::move(x->right);
    x->right = std::move(y->left);
    Refresh(x);
    y->left = std::move(x);
    Refresh(y);
    return y;
  }
  static std::unique_ptr<Node> Rebalance(std::unique_ptr<Node> n) {
    Refresh(n);
    int bf = Balance(n);
    if (bf > 1) {
      if (Balance(n->left) < 0) {
        n->left = RotateLeft(std::move(n->left));
      }
      return RotateRight(std::move(n));
    }
    if (bf < -1) {
      if (Balance(n->right) > 0) {
        n->right = RotateRight(std::move(n->right));
      }
      return RotateLeft(std::move(n));
    }
    return n;
  }

  // BST insert keyed by `start` (ties allowed, go right).
  static std::unique_ptr<Node> InsertImpl(std::unique_ptr<Node> n,
                                          std::size_t s, std::size_t e,
                                          V v) {
    if (!n) return std::make_unique<Node>(s, e, std::move(v));
    if (s < n->start) {
      n->left = InsertImpl(std::move(n->left), s, e, std::move(v));
    } else {
      n->right = InsertImpl(std::move(n->right), s, e, std::move(v));
    }
    return Rebalance(std::move(n));
  }

  // Erase a single node matching (start, end). If multiple match, erases
  // one (the first found by tree walk). Returns whether anything was erased.
  static std::pair<std::unique_ptr<Node>, bool> EraseOne(
      std::unique_ptr<Node> n, std::size_t s, std::size_t e) {
    if (!n) return {nullptr, false};
    bool erased = false;
    if (s < n->start) {
      auto [new_left, e_l] = EraseOne(std::move(n->left), s, e);
      n->left = std::move(new_left);
      erased = e_l;
    } else if (s > n->start) {
      auto [new_right, e_r] = EraseOne(std::move(n->right), s, e);
      n->right = std::move(new_right);
      erased = e_r;
    } else if (n->end == e) {
      // hit
      if (!n->left) return {std::move(n->right), true};
      if (!n->right) return {std::move(n->left), true};
      // two children: replace with in-order successor (min of right subtree)
      Node* succ = n->right.get();
      while (succ->left) succ = succ->left.get();
      // splice the successor's data into n, then erase succ from right
      n->start = succ->start;
      n->end = succ->end;
      n->value = std::move(succ->value);
      auto [new_right, _] = EraseOne(std::move(n->right), n->start, n->end);
      n->right = std::move(new_right);
      erased = true;
    } else {
      // same start, different end: try left then right (entries with same
      // start may have been inserted right via the tie-break above).
      auto [new_left, e_l] = EraseOne(std::move(n->left), s, e);
      n->left = std::move(new_left);
      if (e_l) {
        erased = true;
      } else {
        auto [new_right, e_r] = EraseOne(std::move(n->right), s, e);
        n->right = std::move(new_right);
        erased = e_r;
      }
    }
    return {Rebalance(std::move(n)), erased};
  }

  // DFS over overlapping entries with subtree-max-end pruning.
  template <typename Fn>
  static void OverlapsImpl(const std::unique_ptr<Node>& n, std::size_t qs,
                           std::size_t qe, Fn& fn) {
    if (!n || n->subtree_max_end <= qs) return;
    OverlapsImpl(n->left, qs, qe, fn);
    if (n->start >= qe) return;  // right subtree starts even later
    if (qs < n->end) fn(n->start, n->end, n->value);
    OverlapsImpl(n->right, qs, qe, fn);
  }

  std::unique_ptr<Node> root_;
  std::size_t size_ = 0;

 public:
  /// View over overlapping entries, materialised eagerly into a small vector
  /// so callers can iterate with the standard range-for syntax.
  class OverlapRange {
   public:
    [[nodiscard]] auto begin() const { return entries_.begin(); }
    [[nodiscard]] auto end() const { return entries_.end(); }
    [[nodiscard]] bool empty() const { return entries_.empty(); }
    [[nodiscard]] std::size_t size() const { return entries_.size(); }

   private:
    friend class IntervalMap;
    std::vector<Entry> entries_;
  };

  /// Returns the entries whose [start, end) overlaps [query_start,
  /// query_end). Iteration order is unspecified.
  [[nodiscard]] OverlapRange Overlaps(std::size_t query_start,
                                      std::size_t query_end) const {
    OverlapRange out;
    if (!root_) return out;  // common-case fast-path: empty tree.
    auto fn = [&](std::size_t s, std::size_t e, const V& v) {
      out.entries_.push_back(Entry{s, e, v});
    };
    OverlapsImpl(root_, query_start, query_end, fn);
    return out;
  }

  /// Insert (start, end, value). Does not check for overlap with existing
  /// entries.
  void Insert(std::size_t start, std::size_t end, V value) {
    ASSERT_VALID_ARGUMENTS(start < end,
                           "IntervalMap::Insert: empty interval [{}, {})",
                           start, end);
    root_ = InsertImpl(std::move(root_), start, end, std::move(value));
    ++size_;
  }

  /// Trims or erases every entry so that no byte in [range_start, range_end)
  /// remains covered. Entries fully covered are erased; entries straddling
  /// one boundary are trimmed; entries straddling both boundaries split
  /// into two remainders.
  void SupersedeRange(std::size_t range_start, std::size_t range_end) {
    ASSERT_VALID_ARGUMENTS(
        range_start < range_end,
        "IntervalMap::SupersedeRange: empty range [{}, {})", range_start,
        range_end);

    // Collect overlapping entries first.
    std::vector<Entry> overlapping;
    auto fn = [&](std::size_t s, std::size_t e, const V& v) {
      overlapping.push_back(Entry{s, e, v});
    };
    OverlapsImpl(root_, range_start, range_end, fn);

    // Erase each, then re-insert any remnants.
    for (auto& o : overlapping) {
      auto [new_root, erased] =
          EraseOne(std::move(root_), o.start, o.end);
      root_ = std::move(new_root);
      ASSERT_VALID_RUNTIME(erased,
                           "IntervalMap::SupersedeRange: failed to erase "
                           "entry [{}, {}) that should have existed",
                           o.start, o.end);
      --size_;

      if (o.start < range_start) {
        Insert(o.start, range_start, o.value);
      }
      if (range_end < o.end) {
        Insert(range_end, o.end, std::move(o.value));
      }
    }
  }

  void Clear() {
    root_.reset();
    size_ = 0;
  }
  [[nodiscard]] bool Empty() const { return !root_; }
  [[nodiscard]] std::size_t Size() const { return size_; }
};

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
