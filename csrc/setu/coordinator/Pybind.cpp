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
#include "coordinator/datatypes/Pybind.h"

#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "coordinator/Coordinator.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
void InitCoordinatorPybindClass(py::module_& m) {
  py::class_<Coordinator, std::shared_ptr<Coordinator>>(m, "Coordinator")
      .def(py::init<std::size_t>(),
           py::arg("port"),
           "Create a Coordinator with specified port")
      .def("start", &Coordinator::Start, "Start the Coordinator loops")
      .def("stop", &Coordinator::Stop, "Stop the Coordinator loops");
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();

  setu::coordinator::datatypes::InitDatatypesPybindSubmodule(m);
  setu::coordinator::InitCoordinatorPybindClass(m);
}
//==============================================================================
