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
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "metastore/MetaStore.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::NodeId;
using setu::commons::TensorDimName;
using setu::commons::TensorName;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorDimSpec;
using setu::commons::datatypes::TensorShardMetadataPtr;
using setu::commons::datatypes::TensorShardSpec;
using setu::metastore::MetaStore;
//==============================================================================
namespace {
//==============================================================================
// Helper to create a simple 1D tensor shard spec
TensorShardSpec Make1DShardSpec(const TensorName& name, std::size_t total_size,
                                std::int32_t start, std::int32_t end) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("x", total_size, start, end);
  return TensorShardSpec(name, dims, torch::kFloat32, Device(torch::kCPU));
}

// Helper to create a 2D tensor shard spec
TensorShardSpec Make2DShardSpec(const TensorName& name, std::size_t rows,
                                std::size_t cols, std::int32_t row_start,
                                std::int32_t row_end, std::int32_t col_start,
                                std::int32_t col_end) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("row", rows, row_start, row_end);
  dims.emplace_back("col", cols, col_start, col_end);
  return TensorShardSpec(name, dims, torch::kFloat32, Device(torch::kCPU));
}

// Helper to find a dimension by name in a TensorShardSpec
const TensorDimSpec* FindDim(const std::vector<TensorDimSpec>& dims,
                             const TensorDimName& dim_name) {
  for (const auto& dim : dims) {
    if (dim.name == dim_name) {
      return &dim;
    }
  }
  return nullptr;
}
//==============================================================================
}  // namespace
//==============================================================================
// RegisterTensorShard tests
//==============================================================================
TEST(MetaStoreTest, RegisterTensorShard_SingleShard_ReturnsValidMetadata) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register a single shard that covers the entire tensor
  auto spec = Make1DShardSpec(tensor_name, 100, 0, 100);
  auto metadata = store.RegisterTensorShard(spec, owner_node);

  // Verify all TensorShardMetadata fields
  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->spec.name, tensor_name);
  EXPECT_FALSE(metadata->id.is_nil());
  EXPECT_EQ(metadata->spec.dims.size(), 1);
  EXPECT_EQ(metadata->owner, owner_node);

  // Verify dimension name and size
  const auto* dim_x = FindDim(metadata->spec.dims, "x");
  ASSERT_NE(dim_x, nullptr);
  EXPECT_EQ(dim_x->name, "x");
  EXPECT_EQ(dim_x->size, 100);
}

TEST(MetaStoreTest, RegisterTensorShard_MultipleShards_AllRegistered) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();

  // Register two shards that together cover a 1D tensor of size 100
  auto spec1 = Make1DShardSpec(tensor_name, 100, 0, 50);
  auto spec2 = Make1DShardSpec(tensor_name, 100, 50, 100);

  auto metadata1 = store.RegisterTensorShard(spec1, node_0);
  auto metadata2 = store.RegisterTensorShard(spec2, node_1);

  // Verify metadata1 contains expected values
  ASSERT_NE(metadata1, nullptr);
  EXPECT_EQ(metadata1->spec.name, tensor_name);
  EXPECT_FALSE(metadata1->id.is_nil());
  EXPECT_EQ(metadata1->spec.dims.size(), 1);
  const auto* dim1_x = FindDim(metadata1->spec.dims, "x");
  ASSERT_NE(dim1_x, nullptr);
  EXPECT_EQ(dim1_x->name, "x");
  EXPECT_EQ(dim1_x->size, 100);

  // Verify metadata2 contains expected values
  ASSERT_NE(metadata2, nullptr);
  EXPECT_EQ(metadata2->spec.name, tensor_name);
  EXPECT_FALSE(metadata2->id.is_nil());
  EXPECT_EQ(metadata2->spec.dims.size(), 1);
  const auto* dim2_x = FindDim(metadata2->spec.dims, "x");
  ASSERT_NE(dim2_x, nullptr);
  EXPECT_EQ(dim2_x->name, "x");
  EXPECT_EQ(dim2_x->size, 100);

  // Shard IDs should be unique
  EXPECT_NE(metadata1->id, metadata2->id);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 2);
}

TEST(MetaStoreTest, RegisterTensorShard_2DTensor_CorrectDims) {
  MetaStore store;
  const TensorName tensor_name = "matrix";
  const NodeId owner_node = GenerateUUID();

  // Register a 2D shard
  auto spec = Make2DShardSpec(tensor_name, 10, 20, 0, 10, 0, 20);
  auto metadata = store.RegisterTensorShard(spec, owner_node);

  // Verify all TensorShardMetadata fields
  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->spec.name, tensor_name);
  EXPECT_FALSE(metadata->id.is_nil());
  EXPECT_EQ(metadata->spec.dims.size(), 2);

  // Verify row dimension
  const auto* dim_row = FindDim(metadata->spec.dims, "row");
  ASSERT_NE(dim_row, nullptr);
  EXPECT_EQ(dim_row->name, "row");
  EXPECT_EQ(dim_row->size, 10);

  // Verify col dimension
  const auto* dim_col = FindDim(metadata->spec.dims, "col");
  ASSERT_NE(dim_col, nullptr);
  EXPECT_EQ(dim_col->name, "col");
  EXPECT_EQ(dim_col->size, 20);
}
//==============================================================================
// GetNumShardsForTensor tests
//==============================================================================
TEST(MetaStoreTest, GetNumShardsForTensor_UnknownTensor_ReturnsZero) {
  MetaStore store;

  EXPECT_EQ(store.GetNumShardsForTensor("nonexistent"), 0);
}

TEST(MetaStoreTest,
     GetNumShardsForTensor_AfterRegistration_ReturnsCorrectCount) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();
  const NodeId node_2 = GenerateUUID();

  // Register 3 shards and verify each metadata
  auto metadata1 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 0, 30), node_0);
  ASSERT_NE(metadata1, nullptr);
  EXPECT_EQ(metadata1->spec.name, tensor_name);
  EXPECT_FALSE(metadata1->id.is_nil());
  EXPECT_EQ(FindDim(metadata1->spec.dims, "x")->size, 90);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 1);

  auto metadata2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 30, 60), node_1);
  ASSERT_NE(metadata2, nullptr);
  EXPECT_EQ(metadata2->spec.name, tensor_name);
  EXPECT_FALSE(metadata2->id.is_nil());
  EXPECT_EQ(FindDim(metadata2->spec.dims, "x")->size, 90);
  EXPECT_NE(metadata1->id, metadata2->id);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 2);

  auto metadata3 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 60, 90), node_2);
  ASSERT_NE(metadata3, nullptr);
  EXPECT_EQ(metadata3->spec.name, tensor_name);
  EXPECT_FALSE(metadata3->id.is_nil());
  EXPECT_EQ(FindDim(metadata3->spec.dims, "x")->size, 90);
  EXPECT_NE(metadata1->id, metadata3->id);
  EXPECT_NE(metadata2->id, metadata3->id);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 3);
}
//==============================================================================
// AllShardsRegistered tests
//==============================================================================
TEST(MetaStoreTest, AllShardsRegistered_UnknownTensor_ReturnsFalse) {
  MetaStore store;

  EXPECT_FALSE(store.AllShardsRegistered("nonexistent"));
}

TEST(MetaStoreTest, AllShardsRegistered_PartialRegistration_ReturnsFalse) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register only half of the tensor
  auto metadata = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 50), owner_node);
  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->spec.name, tensor_name);
  EXPECT_FALSE(metadata->id.is_nil());

  EXPECT_FALSE(store.AllShardsRegistered(tensor_name));
}

TEST(MetaStoreTest, AllShardsRegistered_FullRegistration_ReturnsTrue) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();

  // Register all shards covering the full tensor
  auto metadata1 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 50), node_0);
  auto metadata2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 50, 100), node_1);
  ASSERT_NE(metadata1, nullptr);
  ASSERT_NE(metadata2, nullptr);
  EXPECT_FALSE(metadata1->id.is_nil());
  EXPECT_FALSE(metadata2->id.is_nil());

  EXPECT_TRUE(store.AllShardsRegistered(tensor_name));
}

TEST(MetaStoreTest, AllShardsRegistered_SingleShardFullTensor_ReturnsTrue) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Single shard covers entire tensor
  auto metadata = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 100), owner_node);
  ASSERT_NE(metadata, nullptr);
  EXPECT_EQ(metadata->spec.name, tensor_name);
  EXPECT_FALSE(metadata->id.is_nil());

  EXPECT_TRUE(store.AllShardsRegistered(tensor_name));
}
//==============================================================================
// GetTensorMetadata tests
//==============================================================================
TEST(MetaStoreTest, GetTensorMetadata_PartialRegistration_ReturnsNullptr) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register only part of the tensor
  auto metadata = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 50), owner_node);
  ASSERT_NE(metadata, nullptr);
  EXPECT_FALSE(metadata->id.is_nil());

  EXPECT_EQ(store.GetTensorMetadata(tensor_name), nullptr);
}

TEST(MetaStoreTest, GetTensorMetadata_UnknownTensor_ReturnsNullptr) {
  MetaStore store;

  EXPECT_EQ(store.GetTensorMetadata("nonexistent"), nullptr);
}

TEST(MetaStoreTest, GetTensorMetadata_FullRegistration_ReturnsValidMetadata) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();

  // Register all shards
  auto shard_metadata1 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 50), node_0);
  auto shard_metadata2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 50, 100), node_1);
  ASSERT_NE(shard_metadata1, nullptr);
  ASSERT_NE(shard_metadata2, nullptr);
  EXPECT_FALSE(shard_metadata1->id.is_nil());
  EXPECT_FALSE(shard_metadata2->id.is_nil());

  auto tensor_metadata = store.GetTensorMetadata(tensor_name);

  ASSERT_NE(tensor_metadata, nullptr);
  EXPECT_EQ(tensor_metadata->name, tensor_name);
  EXPECT_EQ(tensor_metadata->size, 100);
  EXPECT_EQ(tensor_metadata->shards.size(), 2);
  EXPECT_EQ(tensor_metadata->dtype, torch::kFloat32);
}

TEST(MetaStoreTest, GetTensorMetadata_CachesResult_ReturnsSamePointer) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  auto shard_metadata = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 100), owner_node);
  ASSERT_NE(shard_metadata, nullptr);
  EXPECT_FALSE(shard_metadata->id.is_nil());

  auto tensor_metadata1 = store.GetTensorMetadata(tensor_name);
  auto tensor_metadata2 = store.GetTensorMetadata(tensor_name);

  // Should return the same cached pointer
  EXPECT_EQ(tensor_metadata1.get(), tensor_metadata2.get());
}

TEST(MetaStoreTest, GetTensorMetadata_2DTensor_CorrectMetadata) {
  MetaStore store;
  const TensorName tensor_name = "matrix";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();
  const NodeId node_2 = GenerateUUID();
  const NodeId node_3 = GenerateUUID();

  // Register 4 shards for a 10x20 matrix (2x2 grid of shards)
  auto shard1 = store.RegisterTensorShard(
      Make2DShardSpec(tensor_name, 10, 20, 0, 5, 0, 10), node_0);
  auto shard2 = store.RegisterTensorShard(
      Make2DShardSpec(tensor_name, 10, 20, 0, 5, 10, 20), node_1);
  auto shard3 = store.RegisterTensorShard(
      Make2DShardSpec(tensor_name, 10, 20, 5, 10, 0, 10), node_2);
  auto shard4 = store.RegisterTensorShard(
      Make2DShardSpec(tensor_name, 10, 20, 5, 10, 10, 20), node_3);

  // Verify all shards have correct tensor name and valid shard IDs
  ASSERT_NE(shard1, nullptr);
  ASSERT_NE(shard2, nullptr);
  ASSERT_NE(shard3, nullptr);
  ASSERT_NE(shard4, nullptr);
  EXPECT_EQ(shard1->spec.name, tensor_name);
  EXPECT_EQ(shard2->spec.name, tensor_name);
  EXPECT_EQ(shard3->spec.name, tensor_name);
  EXPECT_EQ(shard4->spec.name, tensor_name);
  EXPECT_FALSE(shard1->id.is_nil());
  EXPECT_FALSE(shard2->id.is_nil());
  EXPECT_FALSE(shard3->id.is_nil());
  EXPECT_FALSE(shard4->id.is_nil());

  // Verify all shard IDs are unique
  EXPECT_NE(shard1->id, shard2->id);
  EXPECT_NE(shard1->id, shard3->id);
  EXPECT_NE(shard1->id, shard4->id);
  EXPECT_NE(shard2->id, shard3->id);
  EXPECT_NE(shard2->id, shard4->id);
  EXPECT_NE(shard3->id, shard4->id);

  // Verify dimension structure in shard metadata
  EXPECT_EQ(shard1->spec.dims.size(), 2);
  EXPECT_EQ(FindDim(shard1->spec.dims, "row")->size, 10);
  EXPECT_EQ(FindDim(shard1->spec.dims, "col")->size, 20);

  auto tensor_metadata = store.GetTensorMetadata(tensor_name);

  ASSERT_NE(tensor_metadata, nullptr);
  EXPECT_EQ(tensor_metadata->name, tensor_name);
  EXPECT_EQ(tensor_metadata->size, 200);  // 10 * 20
  EXPECT_EQ(tensor_metadata->shards.size(), 4);
  EXPECT_EQ(tensor_metadata->dims.size(), 2);
  EXPECT_EQ(tensor_metadata->dims.at("row").size, 10);
  EXPECT_EQ(tensor_metadata->dims.at("col").size, 20);
}

TEST(MetaStoreTest, GetTensorMetadata_GetOwnerNodeIds_ReturnsAllOwners) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();
  const NodeId node_2 = GenerateUUID();

  auto shard1 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 0, 30), node_0);
  auto shard2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 30, 60), node_1);
  auto shard3 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 60, 90), node_2);
  ASSERT_NE(shard1, nullptr);
  ASSERT_NE(shard2, nullptr);
  ASSERT_NE(shard3, nullptr);
  EXPECT_FALSE(shard1->id.is_nil());
  EXPECT_FALSE(shard2->id.is_nil());
  EXPECT_FALSE(shard3->id.is_nil());

  auto tensor_metadata = store.GetTensorMetadata(tensor_name);
  ASSERT_NE(tensor_metadata, nullptr);

  auto owner_ids = tensor_metadata->GetOwnerNodeIds();
  EXPECT_EQ(owner_ids.size(), 3);
  EXPECT_TRUE(owner_ids.count(node_0) > 0);
  EXPECT_TRUE(owner_ids.count(node_1) > 0);
  EXPECT_TRUE(owner_ids.count(node_2) > 0);
}
//==============================================================================
// Multiple tensors tests
//==============================================================================
TEST(MetaStoreTest, MultipleTensors_IndependentTracking) {
  MetaStore store;
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();

  // Register shards for two different tensors
  auto shard_a = store.RegisterTensorShard(
      Make1DShardSpec("tensor_a", 100, 0, 50), node_0);
  auto shard_b = store.RegisterTensorShard(
      Make1DShardSpec("tensor_b", 200, 0, 200), node_1);

  // Verify shard_a contains correct values for tensor_a
  ASSERT_NE(shard_a, nullptr);
  EXPECT_EQ(shard_a->spec.name, "tensor_a");
  EXPECT_FALSE(shard_a->id.is_nil());
  EXPECT_EQ(shard_a->spec.dims.size(), 1);
  EXPECT_EQ(FindDim(shard_a->spec.dims, "x")->size, 100);

  // Verify shard_b contains correct values for tensor_b
  ASSERT_NE(shard_b, nullptr);
  EXPECT_EQ(shard_b->spec.name, "tensor_b");
  EXPECT_FALSE(shard_b->id.is_nil());
  EXPECT_EQ(shard_b->spec.dims.size(), 1);
  EXPECT_EQ(FindDim(shard_b->spec.dims, "x")->size, 200);

  EXPECT_EQ(store.GetNumShardsForTensor("tensor_a"), 1);
  EXPECT_EQ(store.GetNumShardsForTensor("tensor_b"), 1);

  EXPECT_FALSE(store.AllShardsRegistered("tensor_a"));
  EXPECT_TRUE(store.AllShardsRegistered("tensor_b"));

  EXPECT_EQ(store.GetTensorMetadata("tensor_a"), nullptr);
  ASSERT_NE(store.GetTensorMetadata("tensor_b"), nullptr);
  EXPECT_EQ(store.GetTensorMetadata("tensor_b")->size, 200);
}
//==============================================================================
// Same owner tests
//==============================================================================
TEST(MetaStoreTest,
     RegisterTensorShard_SameOwnerMultipleShards_TracksCorrectly) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId same_owner = GenerateUUID();

  // Same node owns all shards
  auto shard1 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 0, 30), same_owner);
  auto shard2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 30, 60), same_owner);
  auto shard3 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 90, 60, 90), same_owner);

  ASSERT_NE(shard1, nullptr);
  ASSERT_NE(shard2, nullptr);
  ASSERT_NE(shard3, nullptr);

  // Shard IDs should still be unique even with same owner
  EXPECT_NE(shard1->id, shard2->id);
  EXPECT_NE(shard1->id, shard3->id);
  EXPECT_NE(shard2->id, shard3->id);

  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 3);
  EXPECT_TRUE(store.AllShardsRegistered(tensor_name));

  auto tensor_metadata = store.GetTensorMetadata(tensor_name);
  ASSERT_NE(tensor_metadata, nullptr);

  // Only one unique owner
  auto owner_ids = tensor_metadata->GetOwnerNodeIds();
  EXPECT_EQ(owner_ids.size(), 1);
  EXPECT_TRUE(owner_ids.count(same_owner) > 0);
}
//==============================================================================
// Metadata shard ID verification tests
//==============================================================================
TEST(MetaStoreTest, GetTensorMetadata_ShardIdsMatchRegistered) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId node_0 = GenerateUUID();
  const NodeId node_1 = GenerateUUID();

  auto shard_metadata1 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 0, 50), node_0);
  auto shard_metadata2 = store.RegisterTensorShard(
      Make1DShardSpec(tensor_name, 100, 50, 100), node_1);

  ASSERT_NE(shard_metadata1, nullptr);
  ASSERT_NE(shard_metadata2, nullptr);

  auto tensor_metadata = store.GetTensorMetadata(tensor_name);
  ASSERT_NE(tensor_metadata, nullptr);

  // Verify that the shard IDs in tensor metadata match the ones returned during
  // registration
  EXPECT_TRUE(tensor_metadata->shards.count(shard_metadata1->id) > 0);
  EXPECT_TRUE(tensor_metadata->shards.count(shard_metadata2->id) > 0);

  // Verify shard metadata contents
  auto shard1 = tensor_metadata->shards.at(shard_metadata1->id);
  EXPECT_EQ(shard1->owner, node_0);
  EXPECT_EQ(shard1->spec.name, tensor_name);

  auto shard2 = tensor_metadata->shards.at(shard_metadata2->id);
  EXPECT_EQ(shard2->owner, node_1);
  EXPECT_EQ(shard2->spec.name, tensor_name);
}
//==============================================================================
// Different dtype tests
//==============================================================================
TEST(MetaStoreTest, RegisterTensorShard_Float16Dtype_CorrectMetadata) {
  MetaStore store;
  const TensorName tensor_name = "float16_tensor";
  const NodeId owner_node = GenerateUUID();

  // Create shard spec with float16 dtype
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("x", 100, 0, 100);
  TensorShardSpec spec(tensor_name, dims, torch::kFloat16, Device(torch::kCPU));

  auto shard_metadata = store.RegisterTensorShard(spec, owner_node);
  ASSERT_NE(shard_metadata, nullptr);
  EXPECT_EQ(shard_metadata->spec.name, tensor_name);
  EXPECT_FALSE(shard_metadata->id.is_nil());

  auto tensor_metadata = store.GetTensorMetadata(tensor_name);
  ASSERT_NE(tensor_metadata, nullptr);
  EXPECT_EQ(tensor_metadata->dtype, torch::kFloat16);
}
//==============================================================================
// Shard registration validation tests
//==============================================================================
TEST(MetaStoreTest, RegisterTensorShard_DtypeMismatch_ReturnsNullptr) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register first shard with float32
  std::vector<TensorDimSpec> dims1;
  dims1.emplace_back("x", 100, 0, 50);
  TensorShardSpec spec1(tensor_name, dims1, torch::kFloat32,
                        Device(torch::kCPU));
  auto shard1 = store.RegisterTensorShard(spec1, owner_node);
  ASSERT_NE(shard1, nullptr);

  // Attempt to register second shard with float16 - should return nullptr
  std::vector<TensorDimSpec> dims2;
  dims2.emplace_back("x", 100, 50, 100);
  TensorShardSpec spec2(tensor_name, dims2, torch::kFloat16,
                        Device(torch::kCPU));

  auto shard2 = store.RegisterTensorShard(spec2, owner_node);
  EXPECT_EQ(shard2, nullptr);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 1);
}

TEST(MetaStoreTest, RegisterTensorShard_DimensionCountMismatch_ReturnsNullptr) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register first shard with 1 dimension
  auto spec1 = Make1DShardSpec(tensor_name, 100, 0, 50);
  auto shard1 = store.RegisterTensorShard(spec1, owner_node);
  ASSERT_NE(shard1, nullptr);

  // Attempt to register second shard with 2 dimensions - should return nullptr
  auto spec2 = Make2DShardSpec(tensor_name, 100, 20, 50, 100, 0, 20);

  auto shard2 = store.RegisterTensorShard(spec2, owner_node);
  EXPECT_EQ(shard2, nullptr);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 1);
}

TEST(MetaStoreTest, RegisterTensorShard_DimensionNameMismatch_ReturnsNullptr) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register first shard with dimension named "x"
  std::vector<TensorDimSpec> dims1;
  dims1.emplace_back("x", 100, 0, 50);
  TensorShardSpec spec1(tensor_name, dims1, torch::kFloat32,
                        Device(torch::kCPU));
  auto shard1 = store.RegisterTensorShard(spec1, owner_node);
  ASSERT_NE(shard1, nullptr);

  // Attempt to register second shard with dimension named "y" - should return
  // nullptr
  std::vector<TensorDimSpec> dims2;
  dims2.emplace_back("y", 100, 50, 100);
  TensorShardSpec spec2(tensor_name, dims2, torch::kFloat32,
                        Device(torch::kCPU));

  auto shard2 = store.RegisterTensorShard(spec2, owner_node);
  EXPECT_EQ(shard2, nullptr);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 1);
}

TEST(MetaStoreTest, RegisterTensorShard_DimensionSizeMismatch_ReturnsNullptr) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register first shard with dimension size 100
  auto spec1 = Make1DShardSpec(tensor_name, 100, 0, 50);
  auto shard1 = store.RegisterTensorShard(spec1, owner_node);
  ASSERT_NE(shard1, nullptr);

  // Attempt to register second shard with dimension size 200 - should return
  // nullptr
  auto spec2 = Make1DShardSpec(tensor_name, 200, 50, 100);

  auto shard2 = store.RegisterTensorShard(spec2, owner_node);
  EXPECT_EQ(shard2, nullptr);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 1);
}

TEST(MetaStoreTest, RegisterTensorShard_OverlappingShards1D_ReturnsNullptr) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register first shard [0, 60)
  auto spec1 = Make1DShardSpec(tensor_name, 100, 0, 60);
  auto shard1 = store.RegisterTensorShard(spec1, owner_node);
  ASSERT_NE(shard1, nullptr);

  // Attempt to register overlapping shard [40, 100) - should return nullptr
  auto spec2 = Make1DShardSpec(tensor_name, 100, 40, 100);

  auto shard2 = store.RegisterTensorShard(spec2, owner_node);
  EXPECT_EQ(shard2, nullptr);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 1);
}

TEST(MetaStoreTest, RegisterTensorShard_OverlappingShards2D_ReturnsNullptr) {
  MetaStore store;
  const TensorName tensor_name = "matrix";
  const NodeId owner_node = GenerateUUID();

  // Register first shard covering [0,5) x [0,10)
  auto spec1 = Make2DShardSpec(tensor_name, 10, 20, 0, 5, 0, 10);
  auto shard1 = store.RegisterTensorShard(spec1, owner_node);
  ASSERT_NE(shard1, nullptr);

  // Attempt to register overlapping shard [3,8) x [5,15) - overlaps at [3,5) x
  // [5,10)
  auto spec2 = Make2DShardSpec(tensor_name, 10, 20, 3, 8, 5, 15);

  auto shard2 = store.RegisterTensorShard(spec2, owner_node);
  EXPECT_EQ(shard2, nullptr);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 1);
}

TEST(MetaStoreTest, RegisterTensorShard_NonOverlapping2D_Succeeds) {
  MetaStore store;
  const TensorName tensor_name = "matrix";
  const NodeId owner_node = GenerateUUID();

  // Register first shard covering [0,5) x [0,10)
  auto spec1 = Make2DShardSpec(tensor_name, 10, 20, 0, 5, 0, 10);
  auto shard1 = store.RegisterTensorShard(spec1, owner_node);
  ASSERT_NE(shard1, nullptr);

  // Register non-overlapping shard [5,10) x [0,10) - different rows, same cols
  auto spec2 = Make2DShardSpec(tensor_name, 10, 20, 5, 10, 0, 10);
  auto shard2 = store.RegisterTensorShard(spec2, owner_node);
  ASSERT_NE(shard2, nullptr);

  // Register non-overlapping shard [0,5) x [10,20) - same rows, different cols
  auto spec3 = Make2DShardSpec(tensor_name, 10, 20, 0, 5, 10, 20);
  auto shard3 = store.RegisterTensorShard(spec3, owner_node);
  ASSERT_NE(shard3, nullptr);

  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 3);
}

TEST(MetaStoreTest, RegisterTensorShard_AdjacentShards_Succeeds) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register adjacent shards [0,50) and [50,100) - touching but not overlapping
  auto spec1 = Make1DShardSpec(tensor_name, 100, 0, 50);
  auto shard1 = store.RegisterTensorShard(spec1, owner_node);
  ASSERT_NE(shard1, nullptr);

  auto spec2 = Make1DShardSpec(tensor_name, 100, 50, 100);
  auto shard2 = store.RegisterTensorShard(spec2, owner_node);
  ASSERT_NE(shard2, nullptr);

  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 2);
  EXPECT_TRUE(store.AllShardsRegistered(tensor_name));
}

TEST(MetaStoreTest, RegisterTensorShard_IdenticalShard_ReturnsNullptr) {
  MetaStore store;
  const TensorName tensor_name = "test_tensor";
  const NodeId owner_node = GenerateUUID();

  // Register a shard
  auto spec1 = Make1DShardSpec(tensor_name, 100, 0, 50);
  auto shard1 = store.RegisterTensorShard(spec1, owner_node);
  ASSERT_NE(shard1, nullptr);

  // Attempt to register identical shard - should return nullptr (fully
  // overlapping)
  auto spec2 = Make1DShardSpec(tensor_name, 100, 0, 50);

  auto shard2 = store.RegisterTensorShard(spec2, owner_node);
  EXPECT_EQ(shard2, nullptr);
  EXPECT_EQ(store.GetNumShardsForTensor(tensor_name), 1);
}
//==============================================================================
// Metadata dimension verification tests
//==============================================================================
TEST(MetaStoreTest, GetTensorMetadata_DimensionNamesCorrect) {
  MetaStore store;
  const TensorName tensor_name = "named_dims_tensor";
  const NodeId owner_node = GenerateUUID();

  // Create 3D tensor with named dimensions
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("batch", 8, 0, 8);
  dims.emplace_back("sequence", 512, 0, 512);
  dims.emplace_back("hidden", 768, 0, 768);
  TensorShardSpec spec(tensor_name, dims, torch::kFloat32, Device(torch::kCPU));

  auto shard_metadata = store.RegisterTensorShard(spec, owner_node);
  ASSERT_NE(shard_metadata, nullptr);
  EXPECT_EQ(shard_metadata->spec.dims.size(), 3);

  auto tensor_metadata = store.GetTensorMetadata(tensor_name);
  ASSERT_NE(tensor_metadata, nullptr);

  // Verify all dimension names and sizes in tensor metadata
  EXPECT_EQ(tensor_metadata->dims.size(), 3);
  EXPECT_EQ(tensor_metadata->dims.at("batch").name, "batch");
  EXPECT_EQ(tensor_metadata->dims.at("batch").size, 8);
  EXPECT_EQ(tensor_metadata->dims.at("sequence").name, "sequence");
  EXPECT_EQ(tensor_metadata->dims.at("sequence").size, 512);
  EXPECT_EQ(tensor_metadata->dims.at("hidden").name, "hidden");
  EXPECT_EQ(tensor_metadata->dims.at("hidden").size, 768);

  // Verify total size
  EXPECT_EQ(tensor_metadata->size, 8 * 512 * 768);
}
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
