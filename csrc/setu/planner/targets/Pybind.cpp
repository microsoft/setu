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
#include "planner/targets/Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/RegisterSet.h"
#include "planner/targets/backend.h"
#include "planner/targets/nccl.h"
//==============================================================================
namespace setu::planner::targets {
//==============================================================================
void InitTargetsPybind(py::module_& m) {
  using setu::planner::RegisterSet;
  namespace cir = setu::planner::ir::cir;

  py::class_<RegisterSet>(m, "RegisterSet")
      .def(py::init<>(), "Create an empty RegisterSet")
      .def_static("uniform", &RegisterSet::Uniform, py::arg("num_registers"),
                  py::arg("size_bytes"),
                  "Create a uniform RegisterSet where all registers share the "
                  "same size")
      .def("add_register", &RegisterSet::AddRegister, py::arg("size_bytes"),
           "Add a register with the given size; returns the assigned index")
      .def("num_registers", &RegisterSet::NumRegisters,
           "Number of registers in the set")
      .def("empty", &RegisterSet::Empty, "Whether this register set is empty")
      // Pickle support for multiprocessing
      .def(py::pickle(
          [](const RegisterSet& rs) {  // __getstate__
            return rs.sizes_;
          },
          [](std::vector<std::size_t> sizes) {  // __setstate__
            RegisterSet rs;
            for (auto sz : sizes) {
              rs.AddRegister(sz);
            }
            return rs;
          }));

  py::class_<Backend, std::shared_ptr<Backend>>(m, "Backend");

  py::class_<NCCL, Backend, std::shared_ptr<NCCL>>(m, "NCCLBackend")
      .def(py::init<>(), "Create an NCCL backend with no register sets")
      .def(py::init<std::unordered_map<cir::Device, RegisterSet>>(),
           py::arg("register_sets"),
           "Create an NCCL backend with per-device register sets");
}
//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
