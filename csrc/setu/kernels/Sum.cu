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
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "kernels/DispatchUtils.h"
#include "kernels/ReductionUtils.cuh"
//==============================================================================
namespace setu::kernels {
//==============================================================================
/// @brief CUDA kernel for Sum
/// @tparam scalar_t Floating point type (float, half, bfloat16)
/// @param input Input tensor
template<typename scalar_t>
__global__ void SumKernel(
  scalar_t* __restrict__ output /*[out]*/,
  const scalar_t* __restrict__ input /*[in]*/,
  const std::int32_t num_elements /*[in]*/) {
  __shared__ scalar_t s_sum;
  scalar_t sum_val = 0.0f;

  for (std::int32_t idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
    const scalar_t x = static_cast<scalar_t>(input[blockIdx.x * num_elements + idx]);
    sum_val += x;
  }
  sum_val = BlockReduceSum<scalar_t>(sum_val);
  if (threadIdx.x == 0) {
    s_sum = sum_val;
  }
  __syncthreads();

  for (std::int32_t idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
    const scalar_t x = static_cast<scalar_t>(input[blockIdx.x * num_elements + idx]);
    output[blockIdx.x * num_elements + idx] = static_cast<scalar_t>(x * s_sum);
  }
}
//==============================================================================
void Sum(
  torch::Tensor& output /*[out]*/,
  const torch::Tensor& input /*[in]*/) {
  SETU_DISPATCH_FLOATING_TYPES(input.scalar_type(), "Sum", [&] {
    const int num_elements = input.numel();
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    SumKernel<scalar_t><<<num_blocks, block_size>>>(
      output.data_ptr<scalar_t>(),
      input.data_ptr<scalar_t>(),
      num_elements
    );
  });
}
//==============================================================================
}  // namespace setu::kernels
//==============================================================================
