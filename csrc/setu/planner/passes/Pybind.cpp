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
#include "planner/passes/Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/Constants.h"
#include "planner/passes/BandwidthAggregation.h"
#include "planner/passes/InstructionScheduler.h"
#include "planner/passes/PackUnpackCopies.h"
#include "planner/passes/PassManager.h"
#include "planner/passes/Pipelining.h"
#include "planner/passes/RegisterTiling.h"
#include "planner/passes/ShortestPathRouting.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================
using setu::planner::topo::TopologyPtr;
//==============================================================================
void InitPassesPybind(py::module_& m) {
  py::class_<Pass, PassPtr>(m, "Pass");

  py::class_<PassManager, PassManagerPtr>(m, "PassManager")
      .def(py::init<>(), "Create an empty PassManager")
      .def("add_pass", &PassManager::AddPass, py::arg("pass_"),
           "Add a pass to the pipeline")
      .def("num_passes", &PassManager::NumPasses,
           "Return the number of passes in the pipeline");

  py::class_<ShortestPathRouting, Pass, std::shared_ptr<ShortestPathRouting>>(
      m, "ShortestPathRouting")
      .def(py::init<TopologyPtr>(), py::arg("topology").none(true),
           "Create a ShortestPathRouting pass with an optional topology");

  py::class_<PackUnpackCopies, Pass, std::shared_ptr<PackUnpackCopies>>(
      m, "PackUnpackCopies")
      .def(py::init<>(),
           "Create a PackUnpackCopies pass that consolidates cross-device "
           "copies into pack/copy/unpack sequences");

  py::class_<BandwidthAggregation, Pass, std::shared_ptr<BandwidthAggregation>>(
      m, "BandwidthAggregation")
      .def(py::init<TopologyPtr, std::size_t>(), py::arg("topology"),
           py::arg("max_paths") = 4,
           "Create a BandwidthAggregation pass that splits cross-device "
           "copies across multiple edge-disjoint paths");

  py::class_<RegisterTiling, Pass, std::shared_ptr<RegisterTiling>>(
      m, "RegisterTiling")
      .def(py::init<std::size_t>(),
           py::arg("chunk_size_bytes") = setu::planner::kRegisterSize,
           "Create a RegisterTiling pass that splits large AllocTmp buffers "
           "into register-sized chunks");

  py::class_<Pipelining, Pass, std::shared_ptr<Pipelining>>(m, "Pipelining")
      .def(py::init<std::size_t>(), py::arg("chunk_size_elements"),
           "Create a Pipelining pass that splits multi-hop relay chains "
           "into chunks and emits them in wavefront order");

  py::class_<InstructionScheduler, Pass, std::shared_ptr<InstructionScheduler>>(
      m, "InstructionScheduler")
      .def(py::init<>(),
           "Create an InstructionScheduler pass that reorders operations "
           "to minimize AllocTmp register pressure");
}
//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
