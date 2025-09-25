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
#include "commons/utils/CUDAUtils.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
[[nodiscard]] CudaComputeCapability GetCudaComputeCapability(
    torch::Device device /*[in]*/) {
  static std::unordered_map<int, CudaComputeCapability> device_caps;
  int device_idx = device.index();

  auto it = device_caps.find(device_idx);
  if (it != device_caps.end()) {
    return it->second;
  }

  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_idx);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_idx);

  CudaComputeCapability cap{major, minor};
  device_caps[device_idx] = cap;
  return cap;
}
//==============================================================================
[[nodiscard]] int GetCudaRuntimeVersion() {
  static int cached_version = -1;
  if (cached_version != -1) {
    return cached_version;
  }

  cudaRuntimeGetVersion(&cached_version);
  return cached_version;
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
