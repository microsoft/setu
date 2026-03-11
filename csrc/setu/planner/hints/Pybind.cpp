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
  py::class_<PipelineChunkSizeHint>(m, "PipelineChunkSizeHint")
      .def(py::init<std::size_t>(), py::arg("chunk_size_bytes"),
           "Create a hint to override the pipeline chunk size in bytes")
      .def_readonly("chunk_size_bytes",
                    &PipelineChunkSizeHint::chunk_size_bytes,
                    "Chunk size in bytes")
      .def("__repr__", &PipelineChunkSizeHint::ToString)
      .def(py::pickle(
          [](const PipelineChunkSizeHint& h) {  // __getstate__
            return py::make_tuple(h.chunk_size_bytes);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 1) {
              throw std::runtime_error(
                  "Invalid state for PipelineChunkSizeHint");
            }
            return PipelineChunkSizeHint(t[0].cast<std::size_t>());
          }));

  py::enum_<ReplicationStrategy>(m, "ReplicationStrategy")
      .value("AllGather", ReplicationStrategy::kAllGather)
      .value("Naive", ReplicationStrategy::kNaive);

  py::class_<ReplicationHint>(m, "ReplicationHint")
      .def(py::init<setu::commons::TensorName, ReplicationStrategy>(),
           py::arg("dst_name"), py::arg("strategy"),
           "Create a replication hint to control AllGather vs Naive strategy")
      .def_readonly("dst_name", &ReplicationHint::dst_name,
                    "Destination tensor name")
      .def_readonly("strategy", &ReplicationHint::strategy,
                    "Replication strategy")
      .def("__repr__", &ReplicationHint::ToString)
      .def(py::pickle(
          [](const ReplicationHint& rh) {  // __getstate__
            return py::make_tuple(rh.dst_name,
                                  static_cast<std::int32_t>(rh.strategy));
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state for ReplicationHint");
            }
            return ReplicationHint(
                t[0].cast<setu::commons::TensorName>(),
                static_cast<ReplicationStrategy>(t[1].cast<std::int32_t>()));
          }));

  py::class_<RoutingHint>(m, "RoutingHint")
      .def(py::init<Participant, Participant, Path>(), py::arg("src"),
           py::arg("dst"), py::arg("path"),
           "Create a routing hint to override path between src and dst")
      .def_readonly("src", &RoutingHint::src, "Source participant")
      .def_readonly("dst", &RoutingHint::dst, "Destination participant")
      .def_readonly("path", &RoutingHint::path, "Override path")
      .def("__repr__", &RoutingHint::ToString)
      // Pickle support for multiprocessing
      .def(py::pickle(
          [](const RoutingHint& rh) {  // __getstate__
            return py::make_tuple(rh.src, rh.dst, rh.path);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 3) {
              throw std::runtime_error("Invalid state for RoutingHint");
            }
            return RoutingHint(t[0].cast<Participant>(),
                               t[1].cast<Participant>(), t[2].cast<Path>());
          }));

  py::class_<BandwidthHint>(m, "BandwidthHint")
      .def(py::init<Participant, Participant, std::vector<Path>,
                    std::vector<float>>(),
           py::arg("src"), py::arg("dst"), py::arg("paths"), py::arg("weights"),
           "Create a bandwidth hint to override path splitting between src "
           "and dst")
      .def_readonly("src", &BandwidthHint::src, "Source participant")
      .def_readonly("dst", &BandwidthHint::dst, "Destination participant")
      .def_readonly("paths", &BandwidthHint::paths, "Paths to split across")
      .def_readonly("weights", &BandwidthHint::weights,
                    "Fractional weights per path")
      .def("__repr__", &BandwidthHint::ToString)
      .def(py::pickle(
          [](const BandwidthHint& bh) {  // __getstate__
            return py::make_tuple(bh.src, bh.dst, bh.paths, bh.weights);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 4) {
              throw std::runtime_error("Invalid state for BandwidthHint");
            }
            return BandwidthHint(t[0].cast<Participant>(),
                                 t[1].cast<Participant>(),
                                 t[2].cast<std::vector<Path>>(),
                                 t[3].cast<std::vector<float>>());
          }));
}
//==============================================================================
}  // namespace setu::planner::hints
//==============================================================================
