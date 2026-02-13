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
#include "setu/planner/Participant.h"
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
      .def("__repr__", [](const Participant& p) {
        return std::format("Participant(node_id={}, device={})",
                           boost::uuids::to_string(p.node_id),
                           p.device.ToString());
      });
}
//==============================================================================
void InitPlannerPybind(py::module_& m) { InitParticipantPybind(m); }
//==============================================================================
}  // namespace setu::planner
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();
  setu::planner::InitPlannerPybind(m);
}
//==============================================================================
