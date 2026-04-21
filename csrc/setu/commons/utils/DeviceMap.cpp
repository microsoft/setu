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
#include "commons/utils/DeviceMap.h"
//==============================================================================
#include <cuda_runtime.h>
#include <nvml.h>
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
namespace {
//==============================================================================
// Populate `out` for visible CUDA devices with NVML UUIDs as the canonical
// DeviceId (MIG-safe, reboot-safe). NVML is init'd and shutdown in a single
// shot so libnvidia-ml is not attached to the process for its lifetime —
// leaving NVML attached measurably regresses peer-memcpy throughput.
void PopulateFromCuda(DeviceMap& out /*[out]*/) {
  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess || device_count == 0) {
    LOG_DEBUG("DeviceMap::Local: no CUDA devices visible (count={}, status={})",
              device_count, static_cast<int>(cuda_status));
    return;
  }

  // Collect (local_index, pci_bus_id) first so we can do a single NVML
  // init/shutdown around the UUID lookups.
  std::vector<std::pair<int, std::string>> entries;
  entries.reserve(device_count);
  for (int i = 0; i < device_count; ++i) {
    char pci_bus_id[32] = {};
    cuda_status = cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), i);
    ASSERT_VALID_RUNTIME(cuda_status == cudaSuccess,
                         "cudaDeviceGetPCIBusId({}) failed: {}", i,
                         cudaGetErrorString(cuda_status));
    entries.emplace_back(i, std::string(pci_bus_id));
  }

  nvmlReturn_t nvml_status = nvmlInit_v2();
  ASSERT_VALID_RUNTIME(nvml_status == NVML_SUCCESS, "nvmlInit_v2 failed: {}",
                       nvmlErrorString(nvml_status));

  for (const auto& [idx, pci_id] : entries) {
    nvmlDevice_t handle{};
    nvml_status = nvmlDeviceGetHandleByPciBusId_v2(pci_id.c_str(), &handle);
    ASSERT_VALID_RUNTIME(
        nvml_status == NVML_SUCCESS,
        "nvmlDeviceGetHandleByPciBusId_v2('{}') failed: {}", pci_id,
        nvmlErrorString(nvml_status));

    char uuid_buf[NVML_DEVICE_UUID_V2_BUFFER_SIZE] = {};
    nvml_status = nvmlDeviceGetUUID(handle, uuid_buf, sizeof(uuid_buf));
    ASSERT_VALID_RUNTIME(
        nvml_status == NVML_SUCCESS,
        "nvmlDeviceGetUUID failed for cuda:{} ({}): {}", idx, pci_id,
        nvmlErrorString(nvml_status));

    out.Insert(idx, DeviceId(std::string(uuid_buf)));
    LOG_DEBUG("DeviceMap::Local: cuda:{} -> {}", idx, uuid_buf);
  }

  nvml_status = nvmlShutdown();
  ASSERT_VALID_RUNTIME(nvml_status == NVML_SUCCESS, "nvmlShutdown failed: {}",
                       nvmlErrorString(nvml_status));
}
//==============================================================================
}  // namespace
//==============================================================================
void DeviceMap::Insert(int local_index, DeviceId id) {
  ASSERT_VALID_ARGUMENTS(!id.Empty(),
                         "DeviceMap::Insert: empty DeviceId for index {}",
                         local_index);
  auto [it_index, inserted_index] =
      index_to_id_.emplace(local_index, id);
  ASSERT_VALID_RUNTIME(inserted_index,
                       "DeviceMap::Insert: duplicate local index {}",
                       local_index);
  auto [it_id, inserted_id] = id_to_index_.emplace(std::move(id), local_index);
  ASSERT_VALID_RUNTIME(inserted_id,
                       "DeviceMap::Insert: duplicate DeviceId {} for index {}",
                       it_id->first.ToString(), local_index);
}
//==============================================================================
DeviceId DeviceMap::GetDeviceId(int local_index) const {
  auto it = index_to_id_.find(local_index);
  ASSERT_VALID_RUNTIME(it != index_to_id_.end(),
                       "DeviceMap: no DeviceId registered for local index {}",
                       local_index);
  return it->second;
}
//==============================================================================
int DeviceMap::GetLocalIndex(const DeviceId& id) const {
  auto it = id_to_index_.find(id);
  ASSERT_VALID_RUNTIME(
      it != id_to_index_.end(),
      "DeviceMap: DeviceId {} is not visible to this process", id.ToString());
  return it->second;
}
//==============================================================================
const DeviceMap& DeviceMap::Local() {
  static DeviceMap instance;
  static std::once_flag init_flag;
  std::call_once(init_flag, [] { PopulateFromCuda(instance); });
  return instance;
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
