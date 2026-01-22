//==============================================================================
// Copyright 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation
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
#include "Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/enums/Enums.h"
//==============================================================================
namespace setu::commons::enums {
//==============================================================================
void InitDeviceKindPybind(py::module_& m) {
  py::enum_<DeviceKind>(m, "DeviceKind", py::module_local())
      .value("CUDA", DeviceKind::kCuda)
      .value("CPU", DeviceKind::kCpu)
      .value("NVME", DeviceKind::kNvme)
      .export_values();
}
//==============================================================================
void InitDTypePybind(py::module_& m) {
  py::enum_<DType>(m, "DType", py::module_local())
      .value("FLOAT16", DType::kFloat16)
      .value("BFLOAT16", DType::kBFloat16)
      .value("FLOAT32", DType::kFloat32)
      .export_values();
}
//==============================================================================
void InitErrorCodePybind(py::module_& m) {
  py::enum_<ErrorCode>(m, "ErrorCode", py::module_local())
      .value("SUCCESS", ErrorCode::kSuccess)
      .value("INVALID_ARGUMENTS", ErrorCode::kInvalidArguments)
      .value("TIMEOUT", ErrorCode::kTimeout)
      .value("INTERNAL_ERROR", ErrorCode::kInternalError)
      .export_values();
}
//==============================================================================
void InitEnumsPybindSubmodule(py::module_& pm) {
  auto m = pm.def_submodule("enums", "Enums submodule");

  InitDeviceKindPybind(m);
  InitDTypePybind(m);
  InitErrorCodePybind(m);
}
//==============================================================================
}  // namespace setu::commons::enums
//==============================================================================
