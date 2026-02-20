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
#include "planner/hints/Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/hints/Hint.h"
//==============================================================================
namespace setu::planner::hints {
//==============================================================================
void InitHintsPybind(py::module_& m) {
  py::class_<RoutingHint>(m, "RoutingHint")
      .def(py::init<Participant, Participant, Path>(), py::arg("src"),
           py::arg("dst"), py::arg("path"),
           "Create a routing hint to override path between src and dst")
      .def_readonly("src", &RoutingHint::src, "Source participant")
      .def_readonly("dst", &RoutingHint::dst, "Destination participant")
      .def_readonly("path", &RoutingHint::path, "Override path")
      .def("__repr__", &RoutingHint::ToString);
}
//==============================================================================
}  // namespace setu::planner::hints
//==============================================================================
