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
#include <gtest/gtest.h>
//==============================================================================
#include "commons/StdCommon.h"
#include "planner/targets/IntervalMap.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::targets::IntervalMap;
//==============================================================================
namespace {

// Collect values of overlapping entries into a sorted vector for
// order-agnostic comparison.
template <typename V>
std::vector<V> CollectOverlapping(const IntervalMap<V>& map,
                                  std::size_t query_start,
                                  std::size_t query_end) {
  std::vector<V> out;
  for (const auto& entry : map.Overlaps(query_start, query_end)) {
    out.push_back(entry.value);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace
//==============================================================================

TEST(IntervalMapTest, Empty_NoEntries) {
  IntervalMap<std::int32_t> map;
  EXPECT_TRUE(map.Empty());
  EXPECT_EQ(map.Size(), 0u);
  EXPECT_TRUE(CollectOverlapping(map, 0, 100).empty());
}

TEST(IntervalMapTest, Insert_SingleEntry_FoundByOverlap) {
  IntervalMap<std::int32_t> map;
  map.Insert(10, 20, 42);

  EXPECT_FALSE(map.Empty());
  EXPECT_EQ(map.Size(), 1u);
  EXPECT_EQ(CollectOverlapping(map, 0, 100), (std::vector<std::int32_t>{42}));
}

TEST(IntervalMapTest, Overlaps_NonOverlappingQuery_Empty) {
  IntervalMap<std::int32_t> map;
  map.Insert(10, 20, 1);
  EXPECT_TRUE(CollectOverlapping(map, 20, 30).empty()) << "abutting at end";
  EXPECT_TRUE(CollectOverlapping(map, 0, 10).empty()) << "abutting at start";
  EXPECT_TRUE(CollectOverlapping(map, 30, 40).empty()) << "fully after";
}

TEST(IntervalMapTest, Overlaps_PartialEdges_Found) {
  IntervalMap<std::int32_t> map;
  map.Insert(10, 20, 1);

  EXPECT_EQ(CollectOverlapping(map, 5, 11), (std::vector<std::int32_t>{1}))
      << "left-edge overlap";
  EXPECT_EQ(CollectOverlapping(map, 19, 25), (std::vector<std::int32_t>{1}))
      << "right-edge overlap";
  EXPECT_EQ(CollectOverlapping(map, 12, 18), (std::vector<std::int32_t>{1}))
      << "fully inside";
  EXPECT_EQ(CollectOverlapping(map, 0, 100), (std::vector<std::int32_t>{1}))
      << "fully covering";
}

TEST(IntervalMapTest, Overlaps_MultipleEntries_YieldsAll) {
  IntervalMap<std::int32_t> map;
  map.Insert(0, 10, 1);
  map.Insert(15, 25, 2);
  map.Insert(30, 40, 3);

  EXPECT_EQ(CollectOverlapping(map, 0, 100),
            (std::vector<std::int32_t>{1, 2, 3}));
  EXPECT_EQ(CollectOverlapping(map, 5, 20), (std::vector<std::int32_t>{1, 2}));
  EXPECT_EQ(CollectOverlapping(map, 12, 17), (std::vector<std::int32_t>{2}));
}

TEST(IntervalMapTest, SupersedeRange_EraseFullyCovered) {
  IntervalMap<std::int32_t> map;
  map.Insert(10, 20, 1);
  map.Insert(30, 40, 2);

  map.SupersedeRange(5, 45);
  EXPECT_TRUE(map.Empty());
}

TEST(IntervalMapTest, SupersedeRange_TrimLeftEdge) {
  IntervalMap<std::int32_t> map;
  map.Insert(10, 30, 1);

  map.SupersedeRange(5, 15);
  // [10, 30) trimmed to [15, 30)
  auto survivors = map.Overlaps(0, 100);
  auto it = survivors.begin();
  ASSERT_NE(it, survivors.end());
  EXPECT_EQ(it->start, 15u);
  EXPECT_EQ(it->end, 30u);
  EXPECT_EQ(it->value, 1);
  ++it;
  EXPECT_EQ(it, survivors.end());
}

TEST(IntervalMapTest, SupersedeRange_TrimRightEdge) {
  IntervalMap<std::int32_t> map;
  map.Insert(10, 30, 1);

  map.SupersedeRange(25, 40);
  auto survivors = map.Overlaps(0, 100);
  auto it = survivors.begin();
  ASSERT_NE(it, survivors.end());
  EXPECT_EQ(it->start, 10u);
  EXPECT_EQ(it->end, 25u);
  EXPECT_EQ(it->value, 1);
  ++it;
  EXPECT_EQ(it, survivors.end());
}

TEST(IntervalMapTest, SupersedeRange_SplitStraddle) {
  IntervalMap<std::int32_t> map;
  map.Insert(10, 30, 1);

  // Range strictly inside the entry: entry splits into [10,15) and [20,30).
  map.SupersedeRange(15, 20);

  EXPECT_EQ(map.Size(), 2u);
  // Query each half.
  EXPECT_EQ(CollectOverlapping(map, 10, 15), (std::vector<std::int32_t>{1}));
  EXPECT_EQ(CollectOverlapping(map, 20, 30), (std::vector<std::int32_t>{1}));
  // Between the halves, nothing.
  EXPECT_TRUE(CollectOverlapping(map, 15, 20).empty());
}

TEST(IntervalMapTest, SupersedeRange_NoOverlap_Noop) {
  IntervalMap<std::int32_t> map;
  map.Insert(10, 20, 1);
  map.Insert(30, 40, 2);

  map.SupersedeRange(22, 28);
  EXPECT_EQ(map.Size(), 2u);
  EXPECT_EQ(CollectOverlapping(map, 0, 100),
            (std::vector<std::int32_t>{1, 2}));
}

TEST(IntervalMapTest, WriterPattern_SupersedeThenInsert) {
  // Mirrors how the alias DAG uses the map as "latest writer per byte."
  IntervalMap<std::int32_t> map;
  map.Insert(0, 100, 1);       // writer #1 covers [0, 100)
  map.SupersedeRange(40, 60);  // writer #2 takes [40, 60)
  map.Insert(40, 60, 2);

  // Read over [0, 100) should see writers 1 (on [0,40) and [60,100)) and 2.
  EXPECT_EQ(CollectOverlapping(map, 0, 100),
            (std::vector<std::int32_t>{1, 1, 2}));

  // Write-only reads within [40, 60) should see only writer 2.
  EXPECT_EQ(CollectOverlapping(map, 45, 55), (std::vector<std::int32_t>{2}));

  // Write-only reads within [0, 40) should see only writer 1.
  EXPECT_EQ(CollectOverlapping(map, 10, 20), (std::vector<std::int32_t>{1}));
}

TEST(IntervalMapTest, Clear_EmptiesContainer) {
  IntervalMap<std::int32_t> map;
  map.Insert(0, 10, 1);
  map.Insert(20, 30, 2);

  map.Clear();
  EXPECT_TRUE(map.Empty());
  EXPECT_EQ(map.Size(), 0u);
  EXPECT_TRUE(CollectOverlapping(map, 0, 100).empty());
}

TEST(IntervalMapTest, RangeFor_Iteration) {
  // Sanity-check that the iterator works in a range-for loop, since
  // that's the intended call-site shape in the DAG builder.
  IntervalMap<std::int32_t> map;
  map.Insert(0, 10, 1);
  map.Insert(5, 15, 2);

  std::vector<std::int32_t> collected;
  for (const auto& entry : map.Overlaps(3, 7)) {
    collected.push_back(entry.value);
  }
  std::sort(collected.begin(), collected.end());
  EXPECT_EQ(collected, (std::vector<std::int32_t>{1, 2}));
}

//==============================================================================
}  // namespace setu::test::native
//==============================================================================
