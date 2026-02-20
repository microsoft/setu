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
#include "commons/utils/Pybind.h"
#include "node_manager/NodeAgent.h"
#include "node_manager/worker/NCCLWorker.h"
#include "node_manager/worker/Worker.h"
//==============================================================================
namespace setu::node_manager {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::NodeId;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::utils::ZmqContextPtr;
using setu::node_manager::worker::NCCLWorker;
using setu::node_manager::worker::Worker;
//==============================================================================
void InitWorkerPybindClass(py::module_& m) {
  // Worker is abstract (has pure virtual Execute and Setup methods)
  // so we don't provide py::init - it can only be used as a base class
  py::class_<Worker, std::shared_ptr<Worker>>(m, "Worker")
      .def("connect", &Worker::Connect, py::arg("zmq_context"),
           py::arg("endpoint"),
           "Connect the worker to an inproc endpoint on a shared ZMQ context")
      .def("start", &Worker::Start, "Start the worker executor loop")
      .def("stop", &Worker::Stop, "Stop the worker executor loop")
      .def("is_running", &Worker::IsRunning, "Check if worker is running")
      .def_property_readonly("device", &Worker::GetDevice,
                             "Get the device this worker is bound to");

  py::class_<NCCLWorker, Worker, std::shared_ptr<NCCLWorker>>(m, "NCCLWorker")
      .def(py::init<NodeId, Device>(), py::arg("node_id"), py::arg("device"),
           "Create an NCCL worker for the given node ID and device")
      .def("setup", &NCCLWorker::Setup,
           py::call_guard<py::gil_scoped_release>(),
           "Initialize CUDA device and stream (call before execute)")
      .def("execute", &NCCLWorker::Execute, py::arg("program"),
           py::call_guard<py::gil_scoped_release>(),
           "Execute a program (instructions must be embellished with device "
           "pointers)");
}
//==============================================================================
void InitNodeAgentPybindClass(py::module_& m) {
  py::class_<NodeAgent, std::shared_ptr<NodeAgent>>(m, "NodeAgent")
      .def(py::init<NodeId, std::size_t, std::string,
                    const std::vector<Device>&, std::string>(),
           py::arg("node_id"), py::arg("port"), py::arg("coordinator_endpoint"),
           py::arg("devices"), py::arg("lock_base_dir") = NodeAgent::GetDefaultLockBaseDir(),
           "Create a NodeAgent with specified port, coordinator endpoint, "
           "devices, and lock directory")
      .def("start", &NodeAgent::Start, "Start the NodeAgent handler loop")
      .def("stop", &NodeAgent::Stop, "Stop the NodeAgent handler loop");
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
