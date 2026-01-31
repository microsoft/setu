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
#include "commons/datatypes/TensorSelection.h"
#include "commons/utils/ContiguousBufferIterator.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::commons::TensorIndicesBitset;
using setu::commons::TensorIndicesMap;
using setu::commons::datatypes::TensorSelection;
using setu::commons::datatypes::TensorSelectionPtr;
using setu::commons::utils::ContiguousBufferRange;
using setu::commons::utils::ContiguousBufferRangeView;
//==============================================================================
namespace {
//==============================================================================
// Helper to create a bitset from a list of selected indices
TensorIndicesBitset MakeBitset(std::size_t size,
                               const std::vector<std::size_t>& selected) {
  TensorIndicesBitset bitset(size);
  for (auto idx : selected) {
    bitset[idx] = true;
  }
  return bitset;
}

// Helper to create a fully selected bitset
TensorIndicesBitset MakeFullBitset(std::size_t size) {
  TensorIndicesBitset bitset(size);
  bitset.set();
  return bitset;
}

// Helper to collect ranges into a vector for easier testing
std::vector<ContiguousBufferRange> CollectRanges(
    const ContiguousBufferRangeView& view) {
  return std::vector<ContiguousBufferRange>(view.begin(), view.end());
}
//==============================================================================
// Single dimension tests
//==============================================================================
TEST(ContiguousBufferRangeViewTest, SingleDim_FullSelection_SingleRange) {
  // tensor[8] with selection tensor[:]
  // Expected: single range covering all 8 elements
  TensorIndicesMap indices;
  indices["x"] = MakeFullBitset(8);

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"x"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start, 0);
  EXPECT_EQ(ranges[0].length, 8);
}

TEST(ContiguousBufferRangeViewTest, SingleDim_ContiguousSelection_SingleRange) {
  // tensor[8] with selection tensor[2:6]
  // Expected: single range [2, 6)
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(8, {2, 3, 4, 5});

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"x"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start, 2);
  EXPECT_EQ(ranges[0].length, 4);
}

TEST(ContiguousBufferRangeViewTest,
     SingleDim_NonContiguousSelection_MultipleRanges) {
  // tensor[8] with selection tensor[[0,1,5,6,7]]
  // Expected: two ranges [0, 2) and [5, 8)
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(8, {0, 1, 5, 6, 7});

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"x"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 2);
  EXPECT_EQ(ranges[0].start, 0);
  EXPECT_EQ(ranges[0].length, 2);
  EXPECT_EQ(ranges[1].start, 5);
  EXPECT_EQ(ranges[1].length, 3);
}

TEST(ContiguousBufferRangeViewTest, SingleDim_EmptySelection_NoRanges) {
  // tensor[8] with empty selection (nothing selected)
  TensorIndicesMap indices;
  indices["x"] = TensorIndicesBitset(8);  // All zeros

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"x"}, selection);

  EXPECT_TRUE(view.empty());
  EXPECT_EQ(view.size(), 0);
}

TEST(ContiguousBufferRangeViewTest, SingleDim_SingleElement_SingleRange) {
  // tensor[8] with selection tensor[3]
  // Expected: single range [3, 4)
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(8, {3});

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"x"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start, 3);
  EXPECT_EQ(ranges[0].length, 1);
}
//==============================================================================
// Two dimension tests (row-major: row is outer, col is inner)
//==============================================================================
TEST(ContiguousBufferRangeViewTest, TwoDim_FullSelection_SingleRange) {
  // tensor[4, 8] with selection tensor[:, :]
  // Row-major layout: 32 contiguous elements
  TensorIndicesMap indices;
  indices["row"] = MakeFullBitset(4);
  indices["col"] = MakeFullBitset(8);

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"row", "col"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start, 0);
  EXPECT_EQ(ranges[0].length, 32);
}

TEST(ContiguousBufferRangeViewTest, TwoDim_SingleRowFullCols_ContiguousRange) {
  // tensor[4, 8] with selection tensor[1, :]
  // Row-major: row 1 spans elements [8, 16)
  TensorIndicesMap indices;
  indices["row"] = MakeBitset(4, {1});
  indices["col"] = MakeFullBitset(8);

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"row", "col"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start, 8);
  EXPECT_EQ(ranges[0].length, 8);
}

TEST(ContiguousBufferRangeViewTest, TwoDim_ContiguousRowsFullCols_SingleRange) {
  // tensor[4, 8] with selection tensor[1:3, :]
  // Row-major: rows 1-2 span elements [8, 24)
  TensorIndicesMap indices;
  indices["row"] = MakeBitset(4, {1, 2});
  indices["col"] = MakeFullBitset(8);

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"row", "col"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start, 8);
  EXPECT_EQ(ranges[0].length, 16);
}

TEST(ContiguousBufferRangeViewTest,
     TwoDim_NonContiguousRowsFullCols_MultipleRanges) {
  // tensor[4, 8] with selection tensor[[0, 2], :]
  // Row-major: rows 0 and 2 are not adjacent -> 2 separate ranges
  TensorIndicesMap indices;
  indices["row"] = MakeBitset(4, {0, 2});
  indices["col"] = MakeFullBitset(8);

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"row", "col"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 2);
  EXPECT_EQ(ranges[0].start, 0);  // Row 0: [0, 8)
  EXPECT_EQ(ranges[0].length, 8);
  EXPECT_EQ(ranges[1].start, 16);  // Row 2: [16, 24)
  EXPECT_EQ(ranges[1].length, 8);
}

TEST(ContiguousBufferRangeViewTest, TwoDim_AllRowsPartialCols_MultipleRanges) {
  // tensor[4, 8] with selection tensor[:, 2:5]
  // Row-major: each row has a non-contiguous gap -> 4 separate ranges
  TensorIndicesMap indices;
  indices["row"] = MakeFullBitset(4);
  indices["col"] = MakeBitset(8, {2, 3, 4});

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"row", "col"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 4);
  EXPECT_EQ(ranges[0].start, 2);  // Row 0: cols 2-4 -> [2, 5)
  EXPECT_EQ(ranges[0].length, 3);
  EXPECT_EQ(ranges[1].start, 10);  // Row 1: cols 2-4 -> [10, 13)
  EXPECT_EQ(ranges[1].length, 3);
  EXPECT_EQ(ranges[2].start, 18);  // Row 2: cols 2-4 -> [18, 21)
  EXPECT_EQ(ranges[2].length, 3);
  EXPECT_EQ(ranges[3].start, 26);  // Row 3: cols 2-4 -> [26, 29)
  EXPECT_EQ(ranges[3].length, 3);
}

TEST(ContiguousBufferRangeViewTest,
     TwoDim_PartialRowsPartialCols_MultipleRanges) {
  // tensor[4, 8] with selection tensor[1:3, 0:3]
  // Row-major: 2 ranges (one per selected row)
  TensorIndicesMap indices;
  indices["row"] = MakeBitset(4, {1, 2});
  indices["col"] = MakeBitset(8, {0, 1, 2});

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"row", "col"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 2);
  EXPECT_EQ(ranges[0].start, 8);  // Row 1: cols 0-2 -> [8, 11)
  EXPECT_EQ(ranges[0].length, 3);
  EXPECT_EQ(ranges[1].start, 16);  // Row 2: cols 0-2 -> [16, 19)
  EXPECT_EQ(ranges[1].length, 3);
}
//==============================================================================
// Three dimension tests
//==============================================================================
TEST(ContiguousBufferRangeViewTest, ThreeDim_FullSelection_SingleRange) {
  // tensor[2, 3, 4] with selection tensor[:, :, :]
  // Total: 24 contiguous elements
  TensorIndicesMap indices;
  indices["batch"] = MakeFullBitset(2);
  indices["row"] = MakeFullBitset(3);
  indices["col"] = MakeFullBitset(4);

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"batch", "row", "col"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start, 0);
  EXPECT_EQ(ranges[0].length, 24);
}

TEST(ContiguousBufferRangeViewTest,
     ThreeDim_SingleBatchFullRowsCols_SingleRange) {
  // tensor[2, 3, 4] with selection tensor[1, :, :]
  // Batch 1 spans elements [12, 24)
  TensorIndicesMap indices;
  indices["batch"] = MakeBitset(2, {1});
  indices["row"] = MakeFullBitset(3);
  indices["col"] = MakeFullBitset(4);

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"batch", "row", "col"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].start, 12);
  EXPECT_EQ(ranges[0].length, 12);
}

TEST(ContiguousBufferRangeViewTest, ThreeDim_AllBatchesSingleRowFullCols) {
  // tensor[2, 3, 4] with selection tensor[:, 1, :]
  // Batch 0, row 1: [4, 8)
  // Batch 1, row 1: [16, 20)
  TensorIndicesMap indices;
  indices["batch"] = MakeFullBitset(2);
  indices["row"] = MakeBitset(3, {1});
  indices["col"] = MakeFullBitset(4);

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"batch", "row", "col"}, selection);

  auto ranges = CollectRanges(view);
  ASSERT_EQ(ranges.size(), 2);
  EXPECT_EQ(ranges[0].start, 4);  // Batch 0, row 1: [4, 8)
  EXPECT_EQ(ranges[0].length, 4);
  EXPECT_EQ(ranges[1].start, 16);  // Batch 1, row 1: [16, 20)
  EXPECT_EQ(ranges[1].length, 4);
}
//==============================================================================
// Edge cases
//==============================================================================
TEST(ContiguousBufferRangeViewTest, EmptyDimNames_NoRanges) {
  TensorIndicesMap indices;
  indices["x"] = MakeFullBitset(8);

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({}, selection);

  EXPECT_TRUE(view.empty());
}

TEST(ContiguousBufferRangeViewTest, RangeBasedForLoop_Works) {
  // tensor[8] with selection tensor[[0,1,4,5,6]]
  // Two ranges: [0, 2) and [4, 7)
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(8, {0, 1, 4, 5, 6});

  auto selection = std::make_shared<TensorSelection>("tensor", indices);
  ContiguousBufferRangeView view({"x"}, selection);

  std::size_t count = 0;
  std::size_t total_length = 0;
  for (const auto& range : view) {
    count++;
    total_length += range.length;
  }

  EXPECT_EQ(count, 2);
  EXPECT_EQ(total_length, 5);  // 2 + 3
}
//==============================================================================
}  // namespace
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
