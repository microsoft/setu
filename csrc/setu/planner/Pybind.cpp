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
#include <boost/uuid/uuid_io.hpp>
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/utils/Pybind.h"
//==============================================================================
#include "planner/Participant.h"
#include "planner/Plan.h"
#include "planner/Planner.h"
#include "planner/ir/llc/Pybind.h"
#include "planner/passes/Pybind.h"
#include "planner/targets/Pybind.h"
#include "planner/topo/Pybind.h"
//==============================================================================
namespace setu::planner {
//==============================================================================
using setu::commons::NodeId;
using setu::commons::datatypes::Device;
//==============================================================================
void InitParticipantPybind(py::module_& m) {
  py::class_<Participant>(m, "Participant")
      .def(py::init<NodeId, Device>(), py::arg("node_id"), py::arg("device"),
           "Create a participant with node ID and device")
      .def_readonly("node_id", &Participant::node_id,
                    "Node ID (UUID) of the participant")
      .def_readonly("device", &Participant::device, "Device of the participant")
      .def("__eq__", &Participant::operator==)
      .def("__lt__", &Participant::operator<)
      .def("__hash__",
           [](const Participant& p) { return std::hash<Participant>{}(p); })
      .def("__repr__",
           [](const Participant& p) {
             return std::format("Participant(node_id={}, device={})",
                                boost::uuids::to_string(p.node_id),
                                p.device.ToString());
           })
      // Pickle support for multiprocessing
      .def(py::pickle(
          [](const Participant& p) {  // __getstate__
            return py::make_tuple(p.node_id, p.device);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state for Participant");
            }
            return Participant(t[0].cast<NodeId>(), t[1].cast<Device>());
          }));
}
//==============================================================================
void InitPlanPybind(py::module_& m) {
  using setu::planner::ir::llc::Program;

  py::class_<Plan>(m, "Plan")
      .def(py::init<>(), "Create an empty Plan")
      .def_readwrite("participants", &Plan::participants,
                     "Set of participants in this plan")
      .def_readwrite("program", &Plan::program,
                     "Mapping from participant to their LLC program")
      .def("fragments", &Plan::Fragments,
           "Split the plan into per-node fragments")
      .def("__str__", &Plan::ToString)
      .def("__repr__", &Plan::ToString);
}
//==============================================================================
void InitPlannerClassPybind(py::module_& m) {
  py::class_<Planner, PlannerPtr>(m, "Planner")
      .def(py::init<targets::BackendPtr, passes::PassManagerPtr>(),
           py::arg("backend"), py::arg("pass_manager"),
           "Create a Planner with the given backend and pass manager")
      .def("compile", &Planner::Compile);
}
//==============================================================================
void InitPlannerPybind(py::module_& m) {
  InitParticipantPybind(m);
  topo::InitTopoPybind(m);
  targets::InitTargetsPybind(m);
  passes::InitPassesPybind(m);
  ir::llc::InitLLCPybind(m);
  InitPlanPybind(m);
  InitPlannerClassPybind(m);
}
//==============================================================================
}  // namespace setu::planner
//==============================================================================
