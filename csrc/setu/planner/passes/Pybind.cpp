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
#include "planner/passes/PassManager.h"
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
      .def(py::init<TopologyPtr>(), py::arg("topology"),
           "Create a ShortestPathRouting pass with the given topology");
}
//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
