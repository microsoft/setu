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
#include "metastore/MetaStore.h"
#include "planner/Plan.h"
#include "planner/Planner.h"
#include "planner/ir/llc/Pybind.h"
#include "planner/targets/nccl.h"
#include "setu/planner/Participant.h"
//==============================================================================
namespace setu::planner {
//==============================================================================
using setu::commons::NodeId;
using setu::commons::datatypes::Device;
using setu::metastore::MetaStore;
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
void InitPlanPybind(py::module_& m) {
  py::class_<Plan>(m, "Plan")
      .def_readonly("participants", &Plan::participants,
                    "Set of participants in the plan")
      .def_readonly("program", &Plan::program,
                    "Mapping from participant to LLC program")
      .def("to_string", &Plan::ToString, "String representation of the plan")
      .def("__str__", &Plan::ToString)
      .def("__repr__", &Plan::ToString);
}
//==============================================================================
void InitMetaStorePybind(py::module_& m) {
  py::class_<MetaStore>(m, "MetaStore")
      .def(py::init<>(), "Create an empty metadata store")
      .def("register_tensor_shard", &MetaStore::RegisterTensorShard,
           py::arg("shard_spec"), py::arg("owner_node_id"),
           "Register a tensor shard and return its metadata")
      .def("all_shards_registered", &MetaStore::AllShardsRegistered,
           py::arg("tensor_name"),
           "Check if all shards for a tensor have been registered")
      .def("get_num_shards_for_tensor", &MetaStore::GetNumShardsForTensor,
           py::arg("tensor_name"),
           "Get number of registered shards for a tensor")
      .def("get_tensor_metadata", &MetaStore::GetTensorMetadata,
           py::arg("tensor_name"),
           "Get tensor metadata (returns None if not fully registered)");
}
//==============================================================================
void InitNCCLPlannerPybind(py::module_& m) {
  py::class_<Planner>(m, "NCCLPlanner")
      .def(py::init([]() {
             return Planner(std::make_unique<targets::NCCL>());
           }),
           "Create a planner with the NCCL backend")
      .def("compile", &Planner::Compile, py::arg("copy_spec"),
           py::arg("metastore"), "Compile a CopySpec into an execution plan");
}
//==============================================================================
void InitPlannerPybind(py::module_& m) {
  InitParticipantPybind(m);
  InitPlanPybind(m);
  InitMetaStorePybind(m);
  InitNCCLPlannerPybind(m);
}
//==============================================================================
}  // namespace setu::planner
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();
  setu::planner::InitPlannerPybind(m);
  setu::planner::ir::llc::InitLLCPybind(m);
}
//==============================================================================
