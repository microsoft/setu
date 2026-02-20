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
#include "planner/topo/Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/topo/Topology.h"
//==============================================================================
namespace setu::planner::topo {
//==============================================================================
void InitTopoPybind(py::module_& m) {
  py::class_<Link>(m, "Link")
      .def(py::init<float, float, std::optional<std::string>>(),
           py::arg("latency_us"), py::arg("bandwidth_gbps"),
           py::arg("tag") = std::nullopt,
           "Create a link with latency and bandwidth")
      // Pickle support for multiprocessing
      .def(py::pickle(
          [](const Link& l) {  // __getstate__
            return py::make_tuple(l.latency_us, l.bandwidth_gbps, l.tag);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 3) {
              throw std::runtime_error("Invalid state for Link");
            }
            return Link(t[0].cast<float>(), t[1].cast<float>(),
                        t[2].cast<std::optional<std::string>>());
          }));

  py::class_<Topology, TopologyPtr>(m, "Topology")
      .def(py::init<>(), "Create an empty topology")
      .def("add_link", &Topology::AddLink, py::arg("src"), py::arg("dst"),
           py::arg("link"), "Add a directed link between participants")
      .def("add_bidirectional_link", &Topology::AddBidirectionalLink,
           py::arg("src"), py::arg("dst"), py::arg("link"),
           "Add a bidirectional link between participants")
      .def("get_edges", &Topology::GetEdges,
           "Get all directed edges as (src, dst, link) tuples")
      // Pickle support for multiprocessing
      .def(py::pickle(
          [](const Topology& t) {  // __getstate__
            return t.GetEdges();
          },
          [](const std::vector<Topology::Edge>& edges) {  // __setstate__
            auto topo = std::make_shared<Topology>();
            for (const auto& [src, dst, link] : edges) {
              topo->AddLink(src, dst, link);
            }
            return topo;
          }));
}
//==============================================================================
}  // namespace setu::planner::topo
//==============================================================================
