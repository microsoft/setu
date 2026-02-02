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
#include "commons/Types.h"
#include "commons/datatypes/TensorDimSpec.h"
#include "commons/datatypes/TensorSelection.h"
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "metastore/datatypes/TensorOwnershipMap.h"
#include "planner/TensorShardRangeView.h"
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::NodeId;
using setu::commons::ShardId;
using setu::commons::TensorDimName;
using setu::commons::TensorIndex;
using setu::commons::TensorIndicesBitset;
using setu::commons::TensorIndicesMap;
using setu::commons::TensorName;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorDimSpec;
using setu::commons::datatypes::TensorSelection;
using setu::commons::datatypes::TensorSelectionPtr;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorShardMetadataMap;
using setu::commons::datatypes::TensorShardMetadataPtr;
using setu::commons::datatypes::TensorShardSpec;
using setu::metastore::datatypes::TensorOwnershipMap;
using setu::metastore::datatypes::TensorOwnershipMapPtr;
using setu::planner::ShardBufferRange;
using setu::planner::TensorShardRangeView;
//==============================================================================

// Helper to create a bitset with specific indices set
TensorIndicesBitset MakeBitset(std::size_t size,
                               const std::vector<std::size_t>& indices) {
  TensorIndicesBitset bitset(size);
  for (auto idx : indices) {
    bitset[idx] = true;
  }
  return bitset;
}

// Helper to create a full bitset
TensorIndicesBitset MakeFullBitset(std::size_t size) {
  TensorIndicesBitset bitset(size);
  bitset.set();
  return bitset;
}

// Helper to create a 1D shard metadata
TensorShardMetadataPtr Make1DShardMetadata(const TensorName& name,
                                           std::size_t size, TensorIndex start,
                                           TensorIndex end) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("x", size, start, end);
  TensorShardSpec spec(name, dims, torch::kFloat32, Device(torch::kCPU));
  return std::make_shared<TensorShardMetadata>(spec, GenerateUUID());
}

// Helper to create a 2D shard metadata
TensorShardMetadataPtr Make2DShardMetadata(const TensorName& name,
                                           std::size_t rows, std::size_t cols,
                                           TensorIndex row_start,
                                           TensorIndex row_end,
                                           TensorIndex col_start,
                                           TensorIndex col_end) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("row", rows, row_start, row_end);
  dims.emplace_back("col", cols, col_start, col_end);
  TensorShardSpec spec(name, dims, torch::kFloat32, Device(torch::kCPU));
  return std::make_shared<TensorShardMetadata>(spec, GenerateUUID());
}

// Helper to create a 3D shard metadata
TensorShardMetadataPtr Make3DShardMetadata(
    const TensorName& name, std::size_t d0, std::size_t d1, std::size_t d2,
    TensorIndex d0_start, TensorIndex d0_end, TensorIndex d1_start,
    TensorIndex d1_end, TensorIndex d2_start, TensorIndex d2_end) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("d0", d0, d0_start, d0_end);
  dims.emplace_back("d1", d1, d1_start, d1_end);
  dims.emplace_back("d2", d2, d2_start, d2_end);
  TensorShardSpec spec(name, dims, torch::kFloat32, Device(torch::kCPU));
  return std::make_shared<TensorShardMetadata>(spec, GenerateUUID());
}

//==============================================================================
// Single Shard Tests
//==============================================================================

TEST(TensorShardRangeViewTest, SingleShard_FullSelection_SingleRange) {
  // 1D tensor of size 10, single shard owns all [0, 10)
  // Selection selects all indices
  // Expected: single range covering entire shard

  const TensorName name = "tensor";
  auto shard = Make1DShardMetadata(name, 10, 0, 10);

  TensorIndicesMap indices;
  indices["x"] = MakeFullBitset(10);
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard->id] = shard;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 1);

  auto it = view.begin();
  EXPECT_EQ(it->metadata->id, shard->id);
  EXPECT_EQ(it->range.start, 0);
  EXPECT_EQ(it->range.length, 10);
}

TEST(TensorShardRangeViewTest, SingleShard_SparseSelection_MultipleRanges) {
  // 1D tensor of size 10, single shard owns all [0, 10)
  // Selection: {0, 1, 2, 5, 6, 9} -> three contiguous ranges
  // Expected: ranges [0,3), [5,7), [9,10)

  const TensorName name = "tensor";
  auto shard = Make1DShardMetadata(name, 10, 0, 10);

  TensorIndicesMap indices;
  indices["x"] = MakeBitset(10, {0, 1, 2, 5, 6, 9});
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard->id] = shard;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 3);

  auto it = view.begin();
  EXPECT_EQ(it->range.start, 0);
  EXPECT_EQ(it->range.length, 3);

  ++it;
  EXPECT_EQ(it->range.start, 5);
  EXPECT_EQ(it->range.length, 2);

  ++it;
  EXPECT_EQ(it->range.start, 9);
  EXPECT_EQ(it->range.length, 1);
}

TEST(TensorShardRangeViewTest, SingleShard_PartialOwnership_LocalizedOffsets) {
  // 1D tensor of size 100, shard owns [25, 50)
  // Selection: {30, 31, 32, 40, 41}
  // Expected: localized to shard space as [5,8) and [15,17)

  const TensorName name = "tensor";
  auto shard = Make1DShardMetadata(name, 100, 25, 50);

  TensorIndicesMap indices;
  indices["x"] = MakeBitset(100, {30, 31, 32, 40, 41});
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard->id] = shard;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 2);

  auto it = view.begin();
  EXPECT_EQ(it->range.start, 5);   // 30 - 25 = 5
  EXPECT_EQ(it->range.length, 3);  // indices 30, 31, 32

  ++it;
  EXPECT_EQ(it->range.start, 15);  // 40 - 25 = 15
  EXPECT_EQ(it->range.length, 2);  // indices 40, 41
}

//==============================================================================
// Two Shard Tests (1D)
//==============================================================================

TEST(TensorShardRangeViewTest, TwoShards_SortedByPosition) {
  // 1D tensor of size 20, two shards:
  // Shard A owns [10, 20) - added first but should come second
  // Shard B owns [0, 10) - added second but should come first
  // Selection: all indices
  // Expected: shard B's range comes before shard A's range

  const TensorName name = "tensor";
  auto shard_a = Make1DShardMetadata(name, 20, 10, 20);
  auto shard_b = Make1DShardMetadata(name, 20, 0, 10);

  TensorIndicesMap indices;
  indices["x"] = MakeFullBitset(20);
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard_a->id] = shard_a;  // Added first
  shards[shard_b->id] = shard_b;  // Added second

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 2);

  auto it = view.begin();
  // First range should be from shard B (position 0)
  EXPECT_EQ(it->metadata->id, shard_b->id);
  EXPECT_EQ(it->range.start, 0);
  EXPECT_EQ(it->range.length, 10);

  ++it;
  // Second range should be from shard A (position 10)
  EXPECT_EQ(it->metadata->id, shard_a->id);
  EXPECT_EQ(it->range.start, 0);  // Localized to shard space
  EXPECT_EQ(it->range.length, 10);
}

TEST(TensorShardRangeViewTest, TwoShards_SelectionSpansBoth) {
  // 1D tensor of size 20
  // Shard A owns [0, 10), Shard B owns [10, 20)
  // Selection: {8, 9, 10, 11} - spans both shards
  // Expected: two ranges, one from each shard

  const TensorName name = "tensor";
  auto shard_a = Make1DShardMetadata(name, 20, 0, 10);
  auto shard_b = Make1DShardMetadata(name, 20, 10, 20);

  TensorIndicesMap indices;
  indices["x"] = MakeBitset(20, {8, 9, 10, 11});
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard_a->id] = shard_a;
  shards[shard_b->id] = shard_b;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 2);

  auto it = view.begin();
  // First: shard A's portion {8, 9}
  EXPECT_EQ(it->metadata->id, shard_a->id);
  EXPECT_EQ(it->range.start, 8);
  EXPECT_EQ(it->range.length, 2);

  ++it;
  // Second: shard B's portion {10, 11} -> localized to {0, 1}
  EXPECT_EQ(it->metadata->id, shard_b->id);
  EXPECT_EQ(it->range.start, 0);
  EXPECT_EQ(it->range.length, 2);
}

//==============================================================================
// 2D Tests - Row Partitioned
//==============================================================================

TEST(TensorShardRangeViewTest, TwoDim_SingleShard_ContiguousRows) {
  // 2D tensor 4x8, single shard owns all
  // Selection: rows {1, 2}, all cols
  // Expected: single contiguous range (rows 1-2 are adjacent in memory)

  const TensorName name = "tensor";
  auto shard = Make2DShardMetadata(name, 4, 8, 0, 4, 0, 8);

  TensorIndicesMap indices;
  indices["row"] = MakeBitset(4, {1, 2});
  indices["col"] = MakeFullBitset(8);
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard->id] = shard;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 1);

  auto it = view.begin();
  EXPECT_EQ(it->range.start, 8);    // row 1 starts at offset 8
  EXPECT_EQ(it->range.length, 16);  // 2 rows * 8 cols
}

TEST(TensorShardRangeViewTest, TwoDim_SingleShard_SparseRows) {
  // 2D tensor 4x8, single shard owns all
  // Selection: rows {0, 2} (non-contiguous), all cols
  // Expected: two ranges (row 0 and row 2 are not adjacent)

  const TensorName name = "tensor";
  auto shard = Make2DShardMetadata(name, 4, 8, 0, 4, 0, 8);

  TensorIndicesMap indices;
  indices["row"] = MakeBitset(4, {0, 2});
  indices["col"] = MakeFullBitset(8);
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard->id] = shard;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 2);

  auto it = view.begin();
  EXPECT_EQ(it->range.start, 0);
  EXPECT_EQ(it->range.length, 8);  // row 0

  ++it;
  EXPECT_EQ(it->range.start, 16);  // row 2 starts at offset 16
  EXPECT_EQ(it->range.length, 8);
}

TEST(TensorShardRangeViewTest, TwoDim_TwoShards_RowPartitioned) {
  // 2D tensor 4x8
  // Shard A owns rows [0, 2), Shard B owns rows [2, 4)
  // Selection: all rows, all cols
  // Expected: two ranges in order (A then B)

  const TensorName name = "tensor";
  auto shard_a = Make2DShardMetadata(name, 4, 8, 0, 2, 0, 8);
  auto shard_b = Make2DShardMetadata(name, 4, 8, 2, 4, 0, 8);

  TensorIndicesMap indices;
  indices["row"] = MakeFullBitset(4);
  indices["col"] = MakeFullBitset(8);
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard_a->id] = shard_a;
  shards[shard_b->id] = shard_b;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 2);

  auto it = view.begin();
  EXPECT_EQ(it->metadata->id, shard_a->id);
  EXPECT_EQ(it->range.start, 0);
  EXPECT_EQ(it->range.length, 16);  // 2 rows * 8 cols

  ++it;
  EXPECT_EQ(it->metadata->id, shard_b->id);
  EXPECT_EQ(it->range.start, 0);    // Localized
  EXPECT_EQ(it->range.length, 16);  // 2 rows * 8 cols
}

//==============================================================================
// 2D Tests - Column Partitioned
//==============================================================================

TEST(TensorShardRangeViewTest, TwoDim_TwoShards_ColPartitioned) {
  // 2D tensor 4x8
  // Shard A owns cols [0, 4), Shard B owns cols [4, 8)
  // Selection: all rows, all cols
  // Expected: shards ordered by row-major position (A before B)

  const TensorName name = "tensor";
  auto shard_a = Make2DShardMetadata(name, 4, 8, 0, 4, 0, 4);
  auto shard_b = Make2DShardMetadata(name, 4, 8, 0, 4, 4, 8);

  TensorIndicesMap indices;
  indices["row"] = MakeFullBitset(4);
  indices["col"] = MakeFullBitset(8);
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard_a->id] = shard_a;
  shards[shard_b->id] = shard_b;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 2);

  auto it = view.begin();
  // Shard A starts at position 0 (row=0, col=0)
  EXPECT_EQ(it->metadata->id, shard_a->id);
  EXPECT_EQ(it->range.start, 0);
  EXPECT_EQ(it->range.length, 16);  // 4 rows * 4 cols

  ++it;
  // Shard B starts at position 4 (row=0, col=4)
  EXPECT_EQ(it->metadata->id, shard_b->id);
  EXPECT_EQ(it->range.start, 0);
  EXPECT_EQ(it->range.length, 16);  // 4 rows * 4 cols
}

TEST(TensorShardRangeViewTest,
     TwoDim_TwoShards_ColPartitioned_SparseSelection) {
  // 2D tensor 4x8
  // Shard A owns cols [0, 4), Shard B owns cols [4, 8)
  // Selection: row {1}, cols {2, 3, 5, 6}
  // Expected: range from A (cols 2,3) then range from B (cols 5,6 -> local 1,2)

  const TensorName name = "tensor";
  auto shard_a = Make2DShardMetadata(name, 4, 8, 0, 4, 0, 4);
  auto shard_b = Make2DShardMetadata(name, 4, 8, 0, 4, 4, 8);

  TensorIndicesMap indices;
  indices["row"] = MakeBitset(4, {1});
  indices["col"] = MakeBitset(8, {2, 3, 5, 6});
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard_a->id] = shard_a;
  shards[shard_b->id] = shard_b;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 2);

  auto it = view.begin();
  EXPECT_EQ(it->metadata->id, shard_a->id);
  // In shard A (4x4): row 1, cols 2,3 -> offset = 1*4 + 2 = 6
  EXPECT_EQ(it->range.start, 6);
  EXPECT_EQ(it->range.length, 2);

  ++it;
  EXPECT_EQ(it->metadata->id, shard_b->id);
  // In shard B (4x4): row 1, cols 1,2 (localized from 5,6) -> offset = 1*4 + 1
  // = 5
  EXPECT_EQ(it->range.start, 5);
  EXPECT_EQ(it->range.length, 2);
}

//==============================================================================
// 2D Tests - 2x2 Grid Partitioning
//==============================================================================

TEST(TensorShardRangeViewTest, TwoDim_FourShards_GridPartitioned) {
  // 2D tensor 4x8, partitioned into 2x2 grid of shards
  // Shard layout (row-major order of shard positions):
  //   [0,2)x[0,4) = position 0   [0,2)x[4,8) = position 4
  //   [2,4)x[0,4) = position 16  [2,4)x[4,8) = position 20
  // Selection: all
  // Expected: shards in row-major order of their start positions

  const TensorName name = "tensor";
  auto shard_00 = Make2DShardMetadata(name, 4, 8, 0, 2, 0, 4);  // pos 0
  auto shard_01 = Make2DShardMetadata(name, 4, 8, 0, 2, 4, 8);  // pos 4
  auto shard_10 = Make2DShardMetadata(name, 4, 8, 2, 4, 0, 4);  // pos 16
  auto shard_11 = Make2DShardMetadata(name, 4, 8, 2, 4, 4, 8);  // pos 20

  TensorIndicesMap indices;
  indices["row"] = MakeFullBitset(4);
  indices["col"] = MakeFullBitset(8);
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  // Add in scrambled order
  shards[shard_11->id] = shard_11;
  shards[shard_00->id] = shard_00;
  shards[shard_10->id] = shard_10;
  shards[shard_01->id] = shard_01;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 4);

  auto it = view.begin();
  EXPECT_EQ(it->metadata->id, shard_00->id);
  EXPECT_EQ(it->range.length, 8);  // 2 rows * 4 cols

  ++it;
  EXPECT_EQ(it->metadata->id, shard_01->id);
  EXPECT_EQ(it->range.length, 8);

  ++it;
  EXPECT_EQ(it->metadata->id, shard_10->id);
  EXPECT_EQ(it->range.length, 8);

  ++it;
  EXPECT_EQ(it->metadata->id, shard_11->id);
  EXPECT_EQ(it->range.length, 8);
}

TEST(TensorShardRangeViewTest, TwoDim_FourShards_SelectionSpansAll) {
  // 2D tensor 4x8, 2x2 grid of shards
  // Selection: rows {1, 2}, cols {3, 4}
  // This selection spans all 4 shards at their boundaries

  const TensorName name = "tensor";
  auto shard_00 = Make2DShardMetadata(name, 4, 8, 0, 2, 0, 4);
  auto shard_01 = Make2DShardMetadata(name, 4, 8, 0, 2, 4, 8);
  auto shard_10 = Make2DShardMetadata(name, 4, 8, 2, 4, 0, 4);
  auto shard_11 = Make2DShardMetadata(name, 4, 8, 2, 4, 4, 8);

  TensorIndicesMap indices;
  indices["row"] = MakeBitset(4, {1, 2});
  indices["col"] = MakeBitset(8, {3, 4});
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard_00->id] = shard_00;
  shards[shard_01->id] = shard_01;
  shards[shard_10->id] = shard_10;
  shards[shard_11->id] = shard_11;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 4);

  auto it = view.begin();
  // shard_00: row 1, col 3 -> local (1, 3), offset = 1*4 + 3 = 7
  EXPECT_EQ(it->metadata->id, shard_00->id);
  EXPECT_EQ(it->range.start, 7);
  EXPECT_EQ(it->range.length, 1);

  ++it;
  // shard_01: row 1, col 4 -> local (1, 0), offset = 1*4 + 0 = 4
  EXPECT_EQ(it->metadata->id, shard_01->id);
  EXPECT_EQ(it->range.start, 4);
  EXPECT_EQ(it->range.length, 1);

  ++it;
  // shard_10: row 2, col 3 -> local (0, 3), offset = 0*4 + 3 = 3
  EXPECT_EQ(it->metadata->id, shard_10->id);
  EXPECT_EQ(it->range.start, 3);
  EXPECT_EQ(it->range.length, 1);

  ++it;
  // shard_11: row 2, col 4 -> local (0, 0), offset = 0
  EXPECT_EQ(it->metadata->id, shard_11->id);
  EXPECT_EQ(it->range.start, 0);
  EXPECT_EQ(it->range.length, 1);
}

//==============================================================================
// 3D Tests
//==============================================================================

TEST(TensorShardRangeViewTest, ThreeDim_SingleShard_FullSelection) {
  // 3D tensor 2x4x8, single shard owns all
  // Selection: all indices
  // Expected: single range covering 64 elements

  const TensorName name = "tensor";
  auto shard = Make3DShardMetadata(name, 2, 4, 8, 0, 2, 0, 4, 0, 8);

  TensorIndicesMap indices;
  indices["d0"] = MakeFullBitset(2);
  indices["d1"] = MakeFullBitset(4);
  indices["d2"] = MakeFullBitset(8);
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard->id] = shard;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 1);

  auto it = view.begin();
  EXPECT_EQ(it->range.start, 0);
  EXPECT_EQ(it->range.length, 64);  // 2 * 4 * 8
}

TEST(TensorShardRangeViewTest, ThreeDim_TwoShards_PartitionedOnD0) {
  // 3D tensor 4x4x8, partitioned on d0
  // Shard A owns d0=[0,2), Shard B owns d0=[2,4)
  // Selection: all indices
  // Expected: A then B

  const TensorName name = "tensor";
  auto shard_a = Make3DShardMetadata(name, 4, 4, 8, 0, 2, 0, 4, 0, 8);
  auto shard_b = Make3DShardMetadata(name, 4, 4, 8, 2, 4, 0, 4, 0, 8);

  TensorIndicesMap indices;
  indices["d0"] = MakeFullBitset(4);
  indices["d1"] = MakeFullBitset(4);
  indices["d2"] = MakeFullBitset(8);
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard_b->id] = shard_b;  // Add B first
  shards[shard_a->id] = shard_a;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 2);

  auto it = view.begin();
  EXPECT_EQ(it->metadata->id, shard_a->id);
  EXPECT_EQ(it->range.length, 64);  // 2 * 4 * 8

  ++it;
  EXPECT_EQ(it->metadata->id, shard_b->id);
  EXPECT_EQ(it->range.length, 64);
}

TEST(TensorShardRangeViewTest, ThreeDim_EightShards_2x2x2Partition) {
  // 3D tensor 4x4x8, partitioned into 2x2x2 = 8 shards
  // Each shard owns a 2x2x4 block
  // Selection: all indices
  // Expected: shards in row-major order

  const TensorName name = "tensor";

  // Create 8 shards in a 2x2x2 grid
  std::vector<TensorShardMetadataPtr> shard_list;
  for (TensorIndex d0 = 0; d0 < 4; d0 += 2) {
    for (TensorIndex d1 = 0; d1 < 4; d1 += 2) {
      for (TensorIndex d2 = 0; d2 < 8; d2 += 4) {
        auto shard = Make3DShardMetadata(name, 4, 4, 8, d0, d0 + 2, d1, d1 + 2,
                                         d2, d2 + 4);
        shard_list.push_back(shard);
      }
    }
  }

  TensorIndicesMap indices;
  indices["d0"] = MakeFullBitset(4);
  indices["d1"] = MakeFullBitset(4);
  indices["d2"] = MakeFullBitset(8);
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  // Add in reverse order to verify sorting works
  for (auto it = shard_list.rbegin(); it != shard_list.rend(); ++it) {
    shards[(*it)->id] = *it;
  }

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 8);

  // Verify each shard has 16 elements (2 * 2 * 4)
  for (const auto& range : view) {
    EXPECT_EQ(range.range.length, 16);
  }

  // Verify ordering by checking row-major positions
  auto it = view.begin();
  std::size_t prev_pos = 0;
  for (std::size_t i = 0; i < 8; ++i, ++it) {
    std::size_t curr_pos = it->metadata->spec.GetRowMajorStartPosition();
    if (i > 0) {
      EXPECT_GT(curr_pos, prev_pos)
          << "Shard " << i << " should come after shard " << (i - 1);
    }
    prev_pos = curr_pos;
  }
}

//==============================================================================
// Edge Cases
//==============================================================================

TEST(TensorShardRangeViewTest, EmptySelection_NoRanges) {
  // Selection has no indices set for one dimension
  // Expected: empty view (no ranges)

  const TensorName name = "tensor";
  auto shard = Make1DShardMetadata(name, 10, 0, 10);

  TensorIndicesMap indices;
  indices["x"] = TensorIndicesBitset(10);  // All zeros
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard->id] = shard;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  EXPECT_TRUE(view.empty());
  EXPECT_EQ(view.size(), 0);
}

TEST(TensorShardRangeViewTest, SelectionOutsideShard_NoRanges) {
  // Shard owns [0, 10), selection is {15, 16, 17}
  // Expected: empty view after localization

  const TensorName name = "tensor";
  auto shard = Make1DShardMetadata(name, 20, 0, 10);

  TensorIndicesMap indices;
  indices["x"] = MakeBitset(20, {15, 16, 17});
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard->id] = shard;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  EXPECT_TRUE(view.empty());
}

TEST(TensorShardRangeViewTest, TwoDim_PartialOverlap_SomeEmpty) {
  // 2D tensor 4x8, 2x2 grid
  // Selection: only row 0, col 0 (single element in top-left shard)
  // Expected: only shard_00 has a range, others empty

  const TensorName name = "tensor";
  auto shard_00 = Make2DShardMetadata(name, 4, 8, 0, 2, 0, 4);
  auto shard_01 = Make2DShardMetadata(name, 4, 8, 0, 2, 4, 8);
  auto shard_10 = Make2DShardMetadata(name, 4, 8, 2, 4, 0, 4);
  auto shard_11 = Make2DShardMetadata(name, 4, 8, 2, 4, 4, 8);

  TensorIndicesMap indices;
  indices["row"] = MakeBitset(4, {0});
  indices["col"] = MakeBitset(8, {0});
  auto selection = std::make_shared<TensorSelection>(name, indices);

  TensorShardMetadataMap shards;
  shards[shard_00->id] = shard_00;
  shards[shard_01->id] = shard_01;
  shards[shard_10->id] = shard_10;
  shards[shard_11->id] = shard_11;

  auto ownership = std::make_shared<TensorOwnershipMap>(selection, shards);
  TensorShardRangeView view(ownership);

  ASSERT_EQ(view.size(), 1);
  EXPECT_EQ(view.begin()->metadata->id, shard_00->id);
  EXPECT_EQ(view.begin()->range.start, 0);
  EXPECT_EQ(view.begin()->range.length, 1);
}

//==============================================================================
