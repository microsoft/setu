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
#include "node_manager/datatypes/Pybind.h"

#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "coordinator/datatypes/Program.h"
#include "node_manager/NodeAgent.h"
#include "node_manager/worker/Worker.h"
//==============================================================================
namespace setu::node_manager {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::NodeRank;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::coordinator::datatypes::Program;
using setu::node_manager::worker::Worker;
//==============================================================================
void InitWorkerPybindClass(py::module_& m) {
  py::class_<Worker, std::shared_ptr<Worker>>(m, "Worker")
      .def(py::init<Device, std::size_t>(), py::arg("device"),
           py::arg("reply_port"),
           "Create a worker bound to a device and reply port")
      .def("start", &Worker::Start, "Start the worker executor loop")
      .def("stop", &Worker::Stop, "Stop the worker executor loop")
      .def("execute", &Worker::Execute, py::arg("program"),
           "Execute a program on the worker")
      .def("is_running", &Worker::IsRunning, "Check if worker is running")
      .def_property_readonly("device", &Worker::GetDevice,
                             "Get the device this worker is bound to");
}
//==============================================================================
void InitNodeAgentPybindClass(py::module_& m) {
  py::class_<NodeAgent, std::shared_ptr<NodeAgent>>(m, "NodeAgent")
      .def(py::init<NodeRank, std::size_t, std::size_t, std::size_t,
                    const std::vector<Device>&>(),
           py::arg("node_rank") = NodeRank{0}, py::arg("router_port"),
           py::arg("dealer_executor_port"), py::arg("dealer_handler_port"),
           py::arg("devices"),
           "Create a NodeAgent with specified ports and devices")
      .def("start", &NodeAgent::Start, "Start the NodeAgent handler loop")
      .def("stop", &NodeAgent::Stop, "Stop the NodeAgent handler loop")
      .def("register_tensor_shard", &NodeAgent::RegisterTensorShard,
           py::arg("shard_spec"),
           "Register a tensor shard and return a reference to it")
      .def("submit_copy", &NodeAgent::SubmitCopy, py::arg("copy_spec"),
           "Submit a copy operation and return an operation ID")
      .def("wait_for_copy", &NodeAgent::WaitForCopy, py::arg("copy_op_id"),
           "Wait for a copy operation to complete")
      .def("allocate_tensor", &NodeAgent::AllocateTensor,
           py::arg("tensor_shars_spec"), "Allocate a tensor shard on a device")
      .def("copy_operation_finished", &NodeAgent::CopyOperationFinished,
           py::arg("copy_op_id"), "Notify that a copy operation has completed")
      .def("execute", &NodeAgent::Execute, py::arg("plan"),
           "Execute a coordinator plan");
}
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();

  setu::node_manager::datatypes::InitDatatypesPybindSubmodule(m);
  setu::node_manager::InitWorkerPybindClass(m);
  setu::node_manager::InitNodeAgentPybindClass(m);
}
//==============================================================================
