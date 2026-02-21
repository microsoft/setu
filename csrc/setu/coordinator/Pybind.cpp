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
#include "planner/Pybind.h"

#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "coordinator/Coordinator.h"
#include "metastore/Pybind.h"
#include "planner/hints/Hint.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::planner::PlannerPtr;
using setu::planner::hints::CompilerHint;
using setu::planner::hints::RoutingHint;
//==============================================================================
void InitCoordinatorPybindClass(py::module_& m) {
  py::class_<Coordinator, std::shared_ptr<Coordinator>>(m, "Coordinator")
      .def(py::init<std::size_t, PlannerPtr>(), py::arg("port"),
           py::arg("planner"),
           "Create a Coordinator with specified port and planner")
      .def("start", &Coordinator::Start, "Start the Coordinator loops")
      .def("stop", &Coordinator::Stop, "Stop the Coordinator loops")
      // TODO: Ideally we'd bind AddHint directly:
      //   .def("add_hint", &Coordinator::AddHint)
      // and let pybind11/stl.h auto-cast RoutingHint -> CompilerHint
      // (std::variant). However, this fails at compile time —
      // pybind11 can't default-construct the type_caster tuple for
      // the member function pointer's arguments. Root cause is
      // unclear, investigate later as this is not a blocking concern.
      // Using a per-type lambda as a workaround; needs a new overload
      // for each hint type added to CompilerHint.
      .def(
          "add_hint",
          [](Coordinator& self, const RoutingHint& hint) {
            self.AddHint(CompilerHint{hint});
          },
          py::arg("hint"), "Add a compiler hint (e.g. RoutingHint)")
      .def("clear_hints", &Coordinator::ClearHints, "Clear all compiler hints");
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();

  setu::metastore::InitMetastorePybind(m);
  setu::planner::InitPlannerPybind(m);
  setu::coordinator::InitCoordinatorPybindClass(m);
}
//==============================================================================
