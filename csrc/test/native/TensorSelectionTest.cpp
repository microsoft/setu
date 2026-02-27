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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/datatypes/Device.h"
#include "commons/datatypes/TensorDimSpec.h"
#include "commons/datatypes/TensorSelection.h"
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/datatypes/TensorShardSpec.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::NodeId;
using setu::commons::TensorIndices;
using setu::commons::TensorIndicesMap;
using setu::commons::TensorName;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorDimSpec;
using setu::commons::datatypes::TensorSelection;
using setu::commons::datatypes::TensorSelectionPtr;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorShardMetadataPtr;
using setu::commons::datatypes::TensorShardSpec;
//==============================================================================
namespace {
//==============================================================================
// Helper to create an IndexRangeSet with specific indices set
TensorIndices MakeBitset(std::size_t size,
                         const std::vector<std::size_t>& selected) {
  std::set<std::int64_t> index_set;
  for (auto idx : selected) {
    index_set.insert(static_cast<std::int64_t>(idx));
  }
  return TensorIndices::FromIndices(size, index_set);
}

// Helper to create a fully selected IndexRangeSet
TensorIndices MakeFullBitset(std::size_t size) {
  return TensorIndices::MakeFull(size);
}

// Helper to create a contiguous range IndexRangeSet [start, end)
TensorIndices MakeRangeBitset(std::size_t size, std::size_t start,
                              std::size_t end) {
  return TensorIndices::MakeSingle(size, static_cast<std::int64_t>(start),
                                   static_cast<std::int64_t>(end));
}

// Helper to create a 1D shard metadata
TensorShardMetadataPtr Make1DShardMetadata(const TensorName& name,
                                           std::size_t total_size,
                                           std::int32_t start,
                                           std::int32_t end) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("x", total_size, start, end);
  TensorShardSpec spec(name, dims, torch::kFloat32, Device(torch::kCPU));
  return std::make_shared<TensorShardMetadata>(spec, GenerateUUID());
}

// Helper to create a 2D shard metadata
TensorShardMetadataPtr Make2DShardMetadata(const TensorName& name,
                                           std::size_t rows, std::size_t cols,
                                           std::int32_t row_start,
                                           std::int32_t row_end,
                                           std::int32_t col_start,
                                           std::int32_t col_end) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("row", rows, row_start, row_end);
  dims.emplace_back("col", cols, col_start, col_end);
  TensorShardSpec spec(name, dims, torch::kFloat32, Device(torch::kCPU));
  return std::make_shared<TensorShardMetadata>(spec, GenerateUUID());
}

// Helper to create a 3D shard metadata
TensorShardMetadataPtr Make3DShardMetadata(
    const TensorName& name, std::size_t batch, std::size_t rows,
    std::size_t cols, std::int32_t batch_start, std::int32_t batch_end,
    std::int32_t row_start, std::int32_t row_end, std::int32_t col_start,
    std::int32_t col_end) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("batch", batch, batch_start, batch_end);
  dims.emplace_back("row", rows, row_start, row_end);
  dims.emplace_back("col", cols, col_start, col_end);
  TensorShardSpec spec(name, dims, torch::kFloat32, Device(torch::kCPU));
  return std::make_shared<TensorShardMetadata>(spec, GenerateUUID());
}

// Helper to get selected indices from an IndexRangeSet as a vector
std::vector<std::size_t> GetSelectedIndices(const TensorIndices& range_set) {
  std::vector<std::size_t> indices;
  for (const auto& r : range_set.ranges) {
    for (std::int64_t i = r.start; i < r.end; ++i) {
      indices.push_back(static_cast<std::size_t>(i));
    }
  }
  return indices;
}
//==============================================================================
}  // namespace
//==============================================================================
// Localize tests - 1D
//==============================================================================
TEST(TensorSelectionLocalizeTest, OneDim_FullSelection_LocalizesToShardSize) {
  // Full tensor: size 100
  // Shard owns [25, 50) - 25 elements
  // Selection: all indices (full tensor)
  // Expected: localized selection has all 25 indices set in size 25

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeFullBitset(100);

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 25, 50);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 25);
  EXPECT_TRUE(local.All());
}

TEST(TensorSelectionLocalizeTest, OneDim_PartialSelection_LocalizesCorrectly) {
  // Full tensor: size 100
  // Shard owns [25, 50)
  // Selection: indices {30, 31, 35, 40} (all within shard)
  // Expected: localized indices {5, 6, 10, 15} in size 25

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(100, {30, 31, 35, 40});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 25, 50);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 25);

  auto selected = GetSelectedIndices(local);
  EXPECT_EQ(selected, std::vector<std::size_t>({5, 6, 10, 15}));
}

TEST(TensorSelectionLocalizeTest,
     OneDim_SelectionPartiallyOutsideShard_OnlyIncludesOverlap) {
  // Full tensor: size 100
  // Shard owns [25, 50)
  // Selection: indices {20, 25, 30, 50, 55} (some outside shard)
  // Expected: only {25, 30} are in shard -> localized {0, 5}

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(100, {20, 25, 30, 50, 55});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 25, 50);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 25);

  auto selected = GetSelectedIndices(local);
  EXPECT_EQ(selected, std::vector<std::size_t>({0, 5}));
}

TEST(TensorSelectionLocalizeTest,
     OneDim_SelectionCompletelyOutsideShard_EmptyResult) {
  // Full tensor: size 100
  // Shard owns [25, 50)
  // Selection: indices {0, 10, 60, 70} (all outside shard)
  // Expected: empty localized selection

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(100, {0, 10, 60, 70});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 25, 50);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 25);
  EXPECT_TRUE(local.None());
}

TEST(TensorSelectionLocalizeTest, OneDim_ShardAtStart_LocalizesCorrectly) {
  // Full tensor: size 100
  // Shard owns [0, 30)
  // Selection: indices {0, 5, 10, 25, 29}
  // Expected: same indices in size 30

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(100, {0, 5, 10, 25, 29});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 0, 30);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 30);

  auto selected = GetSelectedIndices(local);
  EXPECT_EQ(selected, std::vector<std::size_t>({0, 5, 10, 25, 29}));
}

TEST(TensorSelectionLocalizeTest, OneDim_ShardAtEnd_LocalizesCorrectly) {
  // Full tensor: size 100
  // Shard owns [70, 100)
  // Selection: indices {70, 80, 99}
  // Expected: localized indices {0, 10, 29} in size 30

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(100, {70, 80, 99});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 70, 100);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 30);

  auto selected = GetSelectedIndices(local);
  EXPECT_EQ(selected, std::vector<std::size_t>({0, 10, 29}));
}

TEST(TensorSelectionLocalizeTest, OneDim_ContiguousRange_LocalizesCorrectly) {
  // Full tensor: size 100
  // Shard owns [20, 60)
  // Selection: contiguous range [30, 50)
  // Expected: localized range [10, 30) in size 40

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeRangeBitset(100, 30, 50);

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 20, 60);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 40);

  auto selected = GetSelectedIndices(local);
  // Should be contiguous from 10 to 29
  EXPECT_EQ(selected.size(), 20);
  EXPECT_EQ(selected.front(), 10);
  EXPECT_EQ(selected.back(), 29);
}
//==============================================================================
// Localize tests - 2D
//==============================================================================
TEST(TensorSelectionLocalizeTest, TwoDim_FullSelection_LocalizesToShardSize) {
  // Full tensor: 10x20
  // Shard owns rows [2, 5), cols [5, 15) -> 3x10 = 30 elements
  // Selection: all indices
  // Expected: all indices set in 3x10 localized selection

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["row"] = MakeFullBitset(10);
  indices["col"] = MakeFullBitset(20);

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make2DShardMetadata(name, 10, 20, 2, 5, 5, 15);

  auto localized = selection->Localize(shard);

  const auto& row = localized->GetDimIndices("row");
  const auto& col = localized->GetDimIndices("col");

  EXPECT_EQ(row.Size(), 3);
  EXPECT_EQ(col.Size(), 10);
  EXPECT_TRUE(row.All());
  EXPECT_TRUE(col.All());
}

TEST(TensorSelectionLocalizeTest, TwoDim_PartialSelection_LocalizesCorrectly) {
  // Full tensor: 10x20
  // Shard owns rows [2, 5), cols [5, 15)
  // Selection: rows {3, 4}, cols {7, 10, 12}
  // Expected: rows {1, 2}, cols {2, 5, 7} (shifted by shard start)

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["row"] = MakeBitset(10, {3, 4});
  indices["col"] = MakeBitset(20, {7, 10, 12});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make2DShardMetadata(name, 10, 20, 2, 5, 5, 15);

  auto localized = selection->Localize(shard);

  const auto& row = localized->GetDimIndices("row");
  const auto& col = localized->GetDimIndices("col");

  EXPECT_EQ(row.Size(), 3);
  EXPECT_EQ(col.Size(), 10);

  auto row_selected = GetSelectedIndices(row);
  auto col_selected = GetSelectedIndices(col);

  EXPECT_EQ(row_selected, std::vector<std::size_t>({1, 2}));
  EXPECT_EQ(col_selected, std::vector<std::size_t>({2, 5, 7}));
}

TEST(TensorSelectionLocalizeTest, TwoDim_RowsOutsideShard_RowBitsetEmpty) {
  // Full tensor: 10x20
  // Shard owns rows [2, 5), cols [5, 15)
  // Selection: rows {0, 1} (outside shard), cols {7, 10}
  // Expected: row is empty, col has {2, 5}

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["row"] = MakeBitset(10, {0, 1});
  indices["col"] = MakeBitset(20, {7, 10});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make2DShardMetadata(name, 10, 20, 2, 5, 5, 15);

  auto localized = selection->Localize(shard);

  const auto& row = localized->GetDimIndices("row");
  const auto& col = localized->GetDimIndices("col");

  EXPECT_EQ(row.Size(), 3);
  EXPECT_EQ(col.Size(), 10);

  EXPECT_TRUE(row.None());

  auto col_selected = GetSelectedIndices(col);
  EXPECT_EQ(col_selected, std::vector<std::size_t>({2, 5}));
}

TEST(TensorSelectionLocalizeTest, TwoDim_ColsOutsideShard_ColBitsetEmpty) {
  // Full tensor: 10x20
  // Shard owns rows [2, 5), cols [5, 15)
  // Selection: rows {3, 4}, cols {0, 1, 18, 19} (outside shard)
  // Expected: row has {1, 2}, col is empty

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["row"] = MakeBitset(10, {3, 4});
  indices["col"] = MakeBitset(20, {0, 1, 18, 19});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make2DShardMetadata(name, 10, 20, 2, 5, 5, 15);

  auto localized = selection->Localize(shard);

  const auto& row = localized->GetDimIndices("row");
  const auto& col = localized->GetDimIndices("col");

  EXPECT_EQ(row.Size(), 3);
  EXPECT_EQ(col.Size(), 10);

  auto row_selected = GetSelectedIndices(row);
  EXPECT_EQ(row_selected, std::vector<std::size_t>({1, 2}));

  EXPECT_TRUE(col.None());
}

TEST(TensorSelectionLocalizeTest, TwoDim_ContiguousRanges_LocalizesCorrectly) {
  // Full tensor: 8x16
  // Shard owns rows [2, 6), cols [4, 12)
  // Selection: rows [3, 5), cols [6, 10)
  // Expected: rows [1, 3), cols [2, 6) in shard-local coords

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["row"] = MakeRangeBitset(8, 3, 5);
  indices["col"] = MakeRangeBitset(16, 6, 10);

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make2DShardMetadata(name, 8, 16, 2, 6, 4, 12);

  auto localized = selection->Localize(shard);

  const auto& row = localized->GetDimIndices("row");
  const auto& col = localized->GetDimIndices("col");

  EXPECT_EQ(row.Size(), 4);
  EXPECT_EQ(col.Size(), 8);

  auto row_selected = GetSelectedIndices(row);
  auto col_selected = GetSelectedIndices(col);

  EXPECT_EQ(row_selected, std::vector<std::size_t>({1, 2}));
  EXPECT_EQ(col_selected, std::vector<std::size_t>({2, 3, 4, 5}));
}

TEST(TensorSelectionLocalizeTest, TwoDim_ShardAtCorner_LocalizesCorrectly) {
  // Full tensor: 10x10
  // Shard owns top-left corner: rows [0, 5), cols [0, 5)
  // Selection: scattered indices

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["row"] = MakeBitset(10, {0, 2, 4, 7, 9});
  indices["col"] = MakeBitset(10, {1, 3, 4, 8});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make2DShardMetadata(name, 10, 10, 0, 5, 0, 5);

  auto localized = selection->Localize(shard);

  const auto& row = localized->GetDimIndices("row");
  const auto& col = localized->GetDimIndices("col");

  EXPECT_EQ(row.Size(), 5);
  EXPECT_EQ(col.Size(), 5);

  auto row_selected = GetSelectedIndices(row);
  auto col_selected = GetSelectedIndices(col);

  // Only indices within [0, 5) should remain
  EXPECT_EQ(row_selected, std::vector<std::size_t>({0, 2, 4}));
  EXPECT_EQ(col_selected, std::vector<std::size_t>({1, 3, 4}));
}

TEST(TensorSelectionLocalizeTest,
     TwoDim_ShardAtBottomRightCorner_LocalizesCorrectly) {
  // Full tensor: 10x10
  // Shard owns bottom-right corner: rows [5, 10), cols [5, 10)
  // Selection: scattered indices

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["row"] = MakeBitset(10, {0, 2, 5, 7, 9});
  indices["col"] = MakeBitset(10, {1, 5, 6, 9});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make2DShardMetadata(name, 10, 10, 5, 10, 5, 10);

  auto localized = selection->Localize(shard);

  const auto& row = localized->GetDimIndices("row");
  const auto& col = localized->GetDimIndices("col");

  EXPECT_EQ(row.Size(), 5);
  EXPECT_EQ(col.Size(), 5);

  auto row_selected = GetSelectedIndices(row);
  auto col_selected = GetSelectedIndices(col);

  // Indices 5,7,9 -> 0,2,4 in local coords
  EXPECT_EQ(row_selected, std::vector<std::size_t>({0, 2, 4}));
  // Indices 5,6,9 -> 0,1,4 in local coords
  EXPECT_EQ(col_selected, std::vector<std::size_t>({0, 1, 4}));
}
//==============================================================================
// Localize tests - 3D
//==============================================================================
TEST(TensorSelectionLocalizeTest, ThreeDim_FullSelection_LocalizesToShardSize) {
  // Full tensor: 4x8x16
  // Shard owns batch [1, 3), rows [2, 6), cols [4, 12) -> 2x4x8 = 64 elements
  // Selection: all indices
  // Expected: all indices set in 2x4x8 localized selection

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["batch"] = MakeFullBitset(4);
  indices["row"] = MakeFullBitset(8);
  indices["col"] = MakeFullBitset(16);

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make3DShardMetadata(name, 4, 8, 16, 1, 3, 2, 6, 4, 12);

  auto localized = selection->Localize(shard);

  const auto& batch = localized->GetDimIndices("batch");
  const auto& row = localized->GetDimIndices("row");
  const auto& col = localized->GetDimIndices("col");

  EXPECT_EQ(batch.Size(), 2);
  EXPECT_EQ(row.Size(), 4);
  EXPECT_EQ(col.Size(), 8);
  EXPECT_TRUE(batch.All());
  EXPECT_TRUE(row.All());
  EXPECT_TRUE(col.All());
}

TEST(TensorSelectionLocalizeTest,
     ThreeDim_PartialSelection_LocalizesCorrectly) {
  // Full tensor: 4x8x16
  // Shard owns batch [1, 3), rows [2, 6), cols [4, 12)
  // Selection: batch {1, 2}, rows {3, 5}, cols {6, 8, 10}
  // Expected: batch {0, 1}, rows {1, 3}, cols {2, 4, 6}

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["batch"] = MakeBitset(4, {1, 2});
  indices["row"] = MakeBitset(8, {3, 5});
  indices["col"] = MakeBitset(16, {6, 8, 10});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make3DShardMetadata(name, 4, 8, 16, 1, 3, 2, 6, 4, 12);

  auto localized = selection->Localize(shard);

  auto batch_selected = GetSelectedIndices(localized->GetDimIndices("batch"));
  auto row_selected = GetSelectedIndices(localized->GetDimIndices("row"));
  auto col_selected = GetSelectedIndices(localized->GetDimIndices("col"));

  EXPECT_EQ(batch_selected, std::vector<std::size_t>({0, 1}));
  EXPECT_EQ(row_selected, std::vector<std::size_t>({1, 3}));
  EXPECT_EQ(col_selected, std::vector<std::size_t>({2, 4, 6}));
}

TEST(TensorSelectionLocalizeTest, ThreeDim_OneDimOutsideShard_ThatDimEmpty) {
  // Full tensor: 4x8x16
  // Shard owns batch [1, 3), rows [2, 6), cols [4, 12)
  // Selection: batch {0, 3} (both outside shard), rows/cols within shard
  // Expected: batch empty

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["batch"] = MakeBitset(4, {0, 3});
  indices["row"] = MakeBitset(8, {3, 4});
  indices["col"] = MakeBitset(16, {6, 8});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make3DShardMetadata(name, 4, 8, 16, 1, 3, 2, 6, 4, 12);

  auto localized = selection->Localize(shard);

  const auto& batch = localized->GetDimIndices("batch");
  EXPECT_TRUE(batch.None());

  auto row_selected = GetSelectedIndices(localized->GetDimIndices("row"));
  auto col_selected = GetSelectedIndices(localized->GetDimIndices("col"));

  EXPECT_EQ(row_selected, std::vector<std::size_t>({1, 2}));
  EXPECT_EQ(col_selected, std::vector<std::size_t>({2, 4}));
}

TEST(TensorSelectionLocalizeTest, ThreeDim_MixedOverlap_LocalizesCorrectly) {
  // Full tensor: 4x8x16
  // Shard owns batch [1, 3), rows [2, 6), cols [4, 12)
  // Selection: batch {0, 1, 2, 3}, rows {0, 3, 7}, cols {2, 6, 14}
  // Expected:
  //   batch: only {1, 2} overlap -> {0, 1}
  //   rows: only {3} overlaps -> {1}
  //   cols: only {6} overlaps -> {2}

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["batch"] = MakeBitset(4, {0, 1, 2, 3});
  indices["row"] = MakeBitset(8, {0, 3, 7});
  indices["col"] = MakeBitset(16, {2, 6, 14});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make3DShardMetadata(name, 4, 8, 16, 1, 3, 2, 6, 4, 12);

  auto localized = selection->Localize(shard);

  auto batch_selected = GetSelectedIndices(localized->GetDimIndices("batch"));
  auto row_selected = GetSelectedIndices(localized->GetDimIndices("row"));
  auto col_selected = GetSelectedIndices(localized->GetDimIndices("col"));

  EXPECT_EQ(batch_selected, std::vector<std::size_t>({0, 1}));
  EXPECT_EQ(row_selected, std::vector<std::size_t>({1}));
  EXPECT_EQ(col_selected, std::vector<std::size_t>({2}));
}

TEST(TensorSelectionLocalizeTest, ThreeDim_ShardCoversEntireTensor_NoShift) {
  // Shard covers entire tensor - localized should match original

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["batch"] = MakeBitset(4, {0, 2});
  indices["row"] = MakeBitset(8, {1, 3, 5});
  indices["col"] = MakeBitset(16, {4, 8, 12});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make3DShardMetadata(name, 4, 8, 16, 0, 4, 0, 8, 0, 16);

  auto localized = selection->Localize(shard);

  auto batch_selected = GetSelectedIndices(localized->GetDimIndices("batch"));
  auto row_selected = GetSelectedIndices(localized->GetDimIndices("row"));
  auto col_selected = GetSelectedIndices(localized->GetDimIndices("col"));

  EXPECT_EQ(batch_selected, std::vector<std::size_t>({0, 2}));
  EXPECT_EQ(row_selected, std::vector<std::size_t>({1, 3, 5}));
  EXPECT_EQ(col_selected, std::vector<std::size_t>({4, 8, 12}));
}
//==============================================================================
// Edge cases
//==============================================================================
TEST(TensorSelectionLocalizeTest, ShardCoversEntireTensor_NoChange) {
  // Shard owns entire tensor
  // Localized selection should have same indices but in same-sized set

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(100, {10, 20, 50, 90});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 0, 100);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 100);

  auto selected = GetSelectedIndices(local);
  EXPECT_EQ(selected, std::vector<std::size_t>({10, 20, 50, 90}));
}

TEST(TensorSelectionLocalizeTest, SingleElementShard_LocalizesCorrectly) {
  // Shard owns just one element [50, 51)
  // Selection includes index 50
  // Expected: size 1 with index 0 set

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(100, {49, 50, 51});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 50, 51);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 1);

  auto selected = GetSelectedIndices(local);
  EXPECT_EQ(selected, std::vector<std::size_t>({0}));
}

TEST(TensorSelectionLocalizeTest,
     SingleElementShard_SelectionDoesNotInclude_Empty) {
  // Shard owns just one element [50, 51)
  // Selection does not include index 50
  // Expected: empty set of size 1

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(100, {49, 51, 52});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 50, 51);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 1);
  EXPECT_TRUE(local.None());
}

TEST(TensorSelectionLocalizeTest, BoundaryIndex_IncludedCorrectly) {
  // Test boundary conditions: shard [25, 50)
  // Index 25 is included (first index of shard)
  // Index 49 is included (last index of shard)
  // Index 24 and 50 are excluded

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeBitset(100, {24, 25, 49, 50});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 25, 50);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 25);

  auto selected = GetSelectedIndices(local);
  // 25 -> 0, 49 -> 24
  EXPECT_EQ(selected, std::vector<std::size_t>({0, 24}));
}

TEST(TensorSelectionLocalizeTest, EmptySelection_LocalizesToEmpty) {
  // Selection has no indices set
  // Localized should also be empty

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = TensorIndices::MakeEmpty(100);

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 100, 25, 50);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 25);
  EXPECT_TRUE(local.None());
}

TEST(TensorSelectionLocalizeTest,
     TwoDim_SingleRowSingleCol_LocalizesCorrectly) {
  // Shard owns single row and single col
  // 10x10 tensor, shard owns row [5, 6), col [5, 6) -> 1 element

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["row"] = MakeBitset(10, {4, 5, 6});
  indices["col"] = MakeBitset(10, {4, 5, 6});

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make2DShardMetadata(name, 10, 10, 5, 6, 5, 6);

  auto localized = selection->Localize(shard);

  const auto& row = localized->GetDimIndices("row");
  const auto& col = localized->GetDimIndices("col");

  EXPECT_EQ(row.Size(), 1);
  EXPECT_EQ(col.Size(), 1);

  auto row_selected = GetSelectedIndices(row);
  auto col_selected = GetSelectedIndices(col);

  EXPECT_EQ(row_selected, std::vector<std::size_t>({0}));
  EXPECT_EQ(col_selected, std::vector<std::size_t>({0}));
}

TEST(TensorSelectionLocalizeTest, LargeTensor_LocalizesCorrectly) {
  // Test with larger dimensions to ensure no overflow issues
  // Tensor: 1000 elements, shard owns [500, 750)

  const TensorName name = "tensor";
  TensorIndicesMap indices;
  indices["x"] = MakeRangeBitset(1000, 600, 700);  // 100 elements

  auto selection = std::make_shared<TensorSelection>(name, indices);
  auto shard = Make1DShardMetadata(name, 1000, 500, 750);

  auto localized = selection->Localize(shard);

  const auto& local = localized->GetDimIndices("x");
  EXPECT_EQ(local.Size(), 250);

  auto selected = GetSelectedIndices(local);
  EXPECT_EQ(selected.size(), 100);
  // 600 -> 100, 699 -> 199
  EXPECT_EQ(selected.front(), 100);
  EXPECT_EQ(selected.back(), 199);
}
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
