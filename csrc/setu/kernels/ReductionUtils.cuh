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
// Adapted from
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/reduce_kernel_utils.cuh
// Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
//==============================================================================
#pragma once
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
// Define shfl_xor_sync helpers for different types
template<typename T>
__inline__ __device__ T shfl_xor_sync_helper(T var, int lane_mask) {
  return __shfl_xor_sync(std::uint32_t(-1), var, lane_mask);
}

// Specialization for c10::Half to resolve ambiguity
template<>
__inline__ __device__ c10::Half shfl_xor_sync_helper<c10::Half>(c10::Half var, int lane_mask) {
  return __shfl_xor_sync(std::uint32_t(-1), static_cast<__half>(var), lane_mask);
}

// Specialization for c10::BFloat16 to resolve ambiguity  
template<>
__inline__ __device__ c10::BFloat16 shfl_xor_sync_helper<c10::BFloat16>(c10::BFloat16 var, int lane_mask) {
  return __shfl_xor_sync(std::uint32_t(-1), static_cast<__nv_bfloat16>(var), lane_mask);
}

#define SETU_SHFL_XOR_SYNC(var, lane_mask) \
    shfl_xor_sync_helper(var, lane_mask)
//==============================================================================
namespace setu::kernels {
//==============================================================================
// Define warp size
constexpr std::int32_t kWarpSize = 32;
//==============================================================================
namespace detail {
//==============================================================================
/// @brief Maximum reduction function for two values
/// @tparam T The numeric type to compare
/// @param a First value
/// @param b Second value
/// @return The maximum of a and b
template <typename T>
__inline__ __device__ T Max(T a /*[in]*/, T b /*[in]*/) {
  return a > b ? a : b;
}

/// @brief Sum reduction function for two values
/// @tparam T The numeric type to add
/// @param a First value
/// @param b Second value
/// @return The sum of a and b
template <typename T>
__inline__ __device__ T Sum(T a /*[in]*/, T b /*[in]*/) {
  return a + b;
}
//==============================================================================
}  // namespace detail
//==============================================================================
/// @brief Function pointer type for reduction operations
/// @tparam T The data type being reduced
template <typename T>
using ReduceFnType = T (*)(T, T);
//==============================================================================
/// @brief Helper function to return the next largest power of 2
/// @param num Input number
/// @return Next power of 2 greater than or equal to num
[[nodiscard]] static constexpr std::int32_t NextPowerOfTwo(std::uint32_t num /*[in]*/) {
  if (num <= 1) return static_cast<std::int32_t>(num);
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}
//==============================================================================
/// @brief Performs a warp-level reduction operation
/// @tparam T The data type being reduced
/// @tparam kNumLanes Number of lanes participating in reduction (must be power of 2)
/// @param val The value to reduce
/// @param fn The reduction function to apply
/// @return The reduced value across the warp
template <typename T, std::int32_t kNumLanes = kWarpSize>
[[nodiscard]] __inline__ __device__ T WarpReduce(T val /*[in]*/, ReduceFnType<T> fn /*[in]*/) {
  static_assert(kNumLanes > 0 && (kNumLanes & (kNumLanes - 1)) == 0,
                "kNumLanes must be a positive power of 2");
  static_assert(kNumLanes <= kWarpSize);
#pragma unroll
  for (std::int32_t mask = kNumLanes >> 1; mask > 0; mask >>= 1) {
    val = fn(val, SETU_SHFL_XOR_SYNC(val, mask));
  }
  return val;
}
//==============================================================================
/// @brief Performs a block-level reduction operation
/// @tparam T The data type being reduced
/// @tparam kMaxBlockSize Maximum block size (must be <= 1024)
/// @param val The value to reduce
/// @param fn The reduction function to apply
/// @return The reduced value across the block
template <typename T, std::int32_t kMaxBlockSize = 1024>
[[nodiscard]] __inline__ __device__ T BlockReduce(T val /*[in]*/, ReduceFnType<T> fn /*[in]*/) {
  static_assert(kMaxBlockSize <= 1024);
  if constexpr (kMaxBlockSize > kWarpSize) {
    val = WarpReduce<T>(val, fn);
    // Calculate max number of lanes that need to participate in the last warp reduce
    constexpr std::int32_t kMaxActiveLanes = (kMaxBlockSize + kWarpSize - 1) / kWarpSize;
    static __shared__ T shared[kMaxActiveLanes];
    const std::int32_t lane = threadIdx.x % kWarpSize;
    const std::int32_t warp_id = threadIdx.x / kWarpSize;
    if (lane == 0) {
      shared[warp_id] = val;
    }

    __syncthreads();

    val = (threadIdx.x < blockDim.x / static_cast<float>(kWarpSize)) ? shared[lane] : static_cast<T>(0.0f);
    val = WarpReduce<T, NextPowerOfTwo(kMaxActiveLanes)>(val, fn);
  } else {
    // A single warp reduce is equal to block reduce
    val = WarpReduce<T, NextPowerOfTwo(kMaxBlockSize)>(val, fn);
  }
  return val;
}
//==============================================================================
/// @brief Performs a block-level maximum reduction
/// @tparam T The numeric type being reduced
/// @tparam kMaxBlockSize Maximum block size (must be <= 1024)
/// @param val The value to reduce
/// @return The maximum value across the block
template <typename T, std::int32_t kMaxBlockSize = 1024>
[[nodiscard]] __inline__ __device__ T BlockReduceMax(T val /*[in]*/) {
  return BlockReduce<T, kMaxBlockSize>(val, detail::Max<T>);
}
//==============================================================================
/// @brief Performs a block-level sum reduction
/// @tparam T The numeric type being reduced
/// @tparam kMaxBlockSize Maximum block size (must be <= 1024)
/// @param val The value to reduce
/// @return The sum of values across the block
template <typename T, std::int32_t kMaxBlockSize = 1024>
[[nodiscard]] __inline__ __device__ T BlockReduceSum(T val /*[in]*/) {
  return BlockReduce<T, kMaxBlockSize>(val, detail::Sum<T>);
}
//==============================================================================
}  // namespace setu::kernels
//==============================================================================
