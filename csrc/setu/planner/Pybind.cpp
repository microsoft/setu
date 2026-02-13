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
#include "planner/Participant.h"
#include "planner/Planner.h"
#include "planner/backends/nccl.h"
//==============================================================================
namespace setu::planner {
//==============================================================================
using setu::commons::NodeId;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorShardSpec;
using setu::metastore::MetaStore;
using setu::planner::backends::nccl::NCCLPlanner;
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
void InitMetaStorePybind(py::module_& m) {
  py::class_<MetaStore>(m, "MetaStore")
      .def(py::init<>(), "Create an empty metadata store")
      .def("register_tensor_shard", &MetaStore::RegisterTensorShard,
           py::arg("shard_spec"), py::arg("owner_node_id"),
           "Register a tensor shard and return its metadata")
      .def("all_shards_registered", &MetaStore::AllShardsRegistered,
           py::arg("tensor_name"),
           "Check if all shards for a tensor are registered")
      .def("get_num_shards_for_tensor", &MetaStore::GetNumShardsForTensor,
           py::arg("tensor_name"),
           "Get number of shards registered for a tensor")
      .def("get_tensor_metadata", &MetaStore::GetTensorMetadata,
           py::arg("tensor_name"),
           "Get tensor metadata if fully registered, None otherwise");
}
//==============================================================================
void InitPlanPybind(py::module_& m) {
  py::class_<Plan>(m, "Plan")
      .def_readonly("participants", &Plan::participants,
                    "Set of participants in this plan")
      .def_readonly("program", &Plan::program,
                    "Map from participant to program (instruction list)")
      .def("to_string", &Plan::ToString, "Get string representation of plan")
      .def("__str__", &Plan::ToString)
      .def("__repr__", &Plan::ToString);
}
//==============================================================================
void InitNCCLPlannerPybind(py::module_& m) {
  py::class_<NCCLPlanner>(m, "NCCLPlanner")
      .def(py::init<>(), "Create an NCCL planner")
      .def(
          "compile",
          [](NCCLPlanner& self, CopySpec& copy_spec, MetaStore& metastore) {
            return self.Compile(copy_spec, metastore);
          },
          py::arg("copy_spec"), py::arg("metastore"),
          "Compile a copy spec into an execution plan");
}
//==============================================================================
void InitPlannerPybind(py::module_& m) {
  InitParticipantPybind(m);
  InitMetaStorePybind(m);
  InitPlanPybind(m);
  InitNCCLPlannerPybind(m);
}
//==============================================================================
}  // namespace setu::planner
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();
  setu::planner::InitPlannerPybind(m);
}
//==============================================================================
