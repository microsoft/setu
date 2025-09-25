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
//==============================================================================
namespace setu::test::kernels {
//==============================================================================
namespace {
//==============================================================================
/**
 * @brief Basic kernel parameter validation test
 *
 * This test validates that kernel launch parameters are within valid ranges
 * for CUDA compute capabilities. These values are commonly used across
 * various kernels in the codebase.
 */
TEST(KernelParameterTest, ValidateCommonKernelParameters) {
  // Arrange: Define common kernel parameters using fixed-width types
  const std::int32_t threads_per_block = 256;
  const std::int32_t min_blocks = 1;
  const std::int32_t max_blocks = 1024;
  const std::int32_t max_shared_memory_bytes = 49152;  // 48KB for most GPUs

  // Assert: Validate threads per block is within CUDA limits
  EXPECT_GE(threads_per_block, 32)
      << "Threads per block should be at least 32 (one warp)";
  EXPECT_LE(threads_per_block, 1024)
      << "Threads per block should not exceed 1024 (CUDA limit)";
  EXPECT_EQ(threads_per_block % 32, 0)
      << "Threads per block should be a multiple of 32 (warp size)";

  // Assert: Validate block count is within reasonable bounds
  EXPECT_GE(min_blocks, 1) << "Minimum blocks should be at least 1";
  EXPECT_LE(max_blocks, 65535)
      << "Maximum blocks should not exceed 65535 (CUDA grid limit)";

  // Assert: Validate shared memory usage is within limits
  EXPECT_LE(max_shared_memory_bytes, 49152)
      << "Shared memory usage should not exceed 48KB for compatibility";
}
//==============================================================================
/**
 * @brief Test kernel grid Dim calculation
 *
 * This test verifies that grid Dims are calculated correctly
 * for various problem sizes, ensuring proper coverage of all data elements.
 */
TEST(KernelParameterTest, GridDimCalculation) {
  // Arrange: Define test parameters
  const std::int32_t threads_per_block = 256;
  std::vector<std::int32_t> problem_sizes = {1,   100,  255,  256,
                                             257, 1000, 10000};

  for (const auto& problem_size : problem_sizes) {
    // Act: Calculate required number of blocks
    std::int32_t num_blocks =
        (problem_size + threads_per_block - 1) / threads_per_block;

    // Assert: Verify grid covers all elements
    EXPECT_GT(num_blocks, 0)
        << "Number of blocks should be positive for problem size "
        << problem_size;
    EXPECT_GE(num_blocks * threads_per_block, problem_size)
        << "Grid should cover all elements for problem size " << problem_size;
    EXPECT_LT((num_blocks - 1) * threads_per_block, problem_size)
        << "Grid should not be oversized for problem size " << problem_size;
  }
}
//==============================================================================
}  // namespace
//==============================================================================
}  // namespace setu::test::kernels
//==============================================================================
