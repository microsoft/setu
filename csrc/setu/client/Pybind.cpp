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
#include "client/Client.h"
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorDim.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/enums/Enums.h"
#include "commons/utils/TorchTensorIPC.h"
//==============================================================================
namespace setu::client {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardRefPtr;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::enums::ErrorCode;
using setu::commons::utils::TensorIPCSpec;
using setu::commons::utils::TensorIPCSpecPtr;
//==============================================================================
void InitClientPybindClass(py::module_& m) {
  py::class_<Client, std::shared_ptr<Client>>(m, "Client")
      .def(py::init<>(), "Create a new client instance")
      .def("connect", &Client::Connect, py::arg("endpoint"),
           "Connect to a NodeAgent at the specified endpoint")
      .def("disconnect", &Client::Disconnect,
           "Disconnect from the current NodeAgent")
      .def("is_connected", &Client::IsConnected,
           "Check if the client is connected")
      .def("get_endpoint", &Client::GetEndpoint,
           "Get the endpoint the client is connected to")
      .def("register_tensor_shard", &Client::RegisterTensorShard,
           py::arg("shard_spec"),
           "Register a tensor shard and return a reference to it")
      .def("submit_copy", &Client::SubmitCopy, py::arg("copy_spec"),
           "Submit a copy operation and return an operation ID")
      .def("wait_for_copy", &Client::WaitForCopy, py::arg("copy_op_id"),
           "Wait for a copy operation to complete")
      .def("get_tensor_handle", &Client::GetTensorHandle,
           py::arg("tensor_name"), "Get the IPC handle for a tensor");
}
//==============================================================================
void InitEnumsPybindClass(py::module_& m) {
  py::enum_<ErrorCode>(m, "ErrorCode")
      .value("SUCCESS", ErrorCode::kSuccess)
      .value("INVALID_ARGUMENTS", ErrorCode::kInvalidArguments)
      .value("TIMEOUT", ErrorCode::kTimeout)
      .value("INTERNAL_ERROR", ErrorCode::kInternalError)
      .value("TENSOR_NOT_FOUND", ErrorCode::kTensorNotFound);
}
//==============================================================================
void InitTensorShardRefPybindClass(py::module_& m) {
  py::class_<TensorShardRef, TensorShardRefPtr>(m, "TensorShardRef",
                                                py::module_local())
      .def(py::init<TensorName, ShardId, TensorDimMap>(), py::arg("name"),
           py::arg("shard_id"), py::arg("dims"))
      .def_readonly("name", &TensorShardRef::name,
                    "Name of the tensor being sharded")
      .def_readonly("shard_id", &TensorShardRef::shard_id,
                    "UUID identifier for this shard")
      .def_readonly("dims", &TensorShardRef::dims,
                    "Map of dimension names to TensorDim objects")
      .def("get_num_dims", &TensorShardRef::GetNumDims,
           "Get number of dimensions in this shard")
      .def("__str__", &TensorShardRef::ToString)
      .def("__repr__", &TensorShardRef::ToString);
}
}  // namespace setu::client
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();

  setu::client::InitEnumsPybindClass(m);
  setu::client::InitTensorShardRefPybindClass(m);
  setu::client::InitClientPybindClass(m);
}
//==============================================================================
