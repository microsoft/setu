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
#include "commons/utils/TorchUtils.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
[[nodiscard]] at::ScalarType GetScalarTypeFromString(
    const std::string& type /*[in]*/) {
  static const std::unordered_map<std::string, at::ScalarType> dtype_map = {
      {"float32", at::ScalarType::Float},
      {"torch.float32", at::ScalarType::Float},
      {"float16", at::ScalarType::Half},
      {"torch.float16", at::ScalarType::Half},
      {"bfloat16", at::ScalarType::BFloat16},
      {"torch.bfloat16", at::ScalarType::BFloat16},
      {"float8_e4m3fn", at::ScalarType::Float8_e4m3fn},
      {"torch.float8_e4m3fn", at::ScalarType::Float8_e4m3fn},
      {"float8_e5m2", at::ScalarType::Float8_e5m2},
      {"torch.float8_e5m2", at::ScalarType::Float8_e5m2},
      {"int8", at::ScalarType::Char},
      {"torch.int8", at::ScalarType::Char},
      {"uint8", at::ScalarType::Byte},
      {"torch.uint8", at::ScalarType::Byte},
      {"int32", at::ScalarType::Int},
      {"torch.int32", at::ScalarType::Int},
      {"uint32", at::ScalarType::UInt32},
      {"torch.uint32", at::ScalarType::UInt32},
      {"int64", at::ScalarType::Long},
      {"torch.int64", at::ScalarType::Long},
      {"uint64", at::ScalarType::UInt64},
      {"torch.uint64", at::ScalarType::UInt64}};

  auto it = dtype_map.find(type);
  ASSERT_VALID_RUNTIME(it != dtype_map.end(), "Invalid dtype: {}.", type);
  return it->second;
}
//==============================================================================
void CheckShapeDtypeAndDevice(
    const torch::Tensor& x /*[in]*/,
    const std::vector<int64_t>& expected_shape /*[in]*/,
    at::ScalarType expected_dtype /*[in]*/,
    torch::Device expected_device /*[in]*/, const std::string& name /*[in]*/
) {
  // Check shape
  auto x_shape = x.sizes().vec();
  ASSERT_VALID_RUNTIME(x_shape == expected_shape,
                       "The shape of {} does not match the expected shape.",
                       name);

  // Check dtype
  ASSERT_VALID_RUNTIME(x.scalar_type() == expected_dtype,
                       "The dtype of {} does not match the expected dtype.",
                       name);

  // Check device
  ASSERT_VALID_RUNTIME(x.device() == expected_device,
                       "The device of {} does not match the expected device.",
                       name);
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
