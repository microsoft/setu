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
#include "Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
#include "commons/Types.h"
#include "commons/messages/Messages.h"
#include "ir/Instruction.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::RequestId;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::enums::ErrorCode;
using setu::ir::Program;
//==============================================================================
void InitRegisterTensorShardRequestPybind(py::module_& m) {
  py::class_<RegisterTensorShardRequest>(m, "RegisterTensorShardRequest",
                                         py::module_local())
      .def(py::init<TensorShardSpec>(), py::arg("tensor_shard_spec"))
      .def_readonly("tensor_shard_spec",
                    &RegisterTensorShardRequest::tensor_shard_spec)
      .def_property_readonly(
          "tensor_name",
          [](const RegisterTensorShardRequest& request) {
            return request.tensor_shard_spec.name;
          },
          "Name of the tensor being registered")
      .def("__str__", &RegisterTensorShardRequest::ToString)
      .def("__repr__", &RegisterTensorShardRequest::ToString);
}
//==============================================================================
void InitRegisterTensorShardResponsePybind(py::module_& m) {
  py::class_<RegisterTensorShardResponse>(m, "RegisterTensorShardResponse",
                                          py::module_local())
      .def(py::init<RequestId, ErrorCode>(), py::arg("request_id"),
           py::arg("error_code") = ErrorCode::kSuccess)
      .def_readonly("request_id", &RegisterTensorShardResponse::request_id)
      .def_readonly("error_code", &RegisterTensorShardResponse::error_code)
      .def("__str__", &RegisterTensorShardResponse::ToString)
      .def("__repr__", &RegisterTensorShardResponse::ToString);
}
//==============================================================================
void InitExecuteProgramRequestPybind(py::module_& m) {
  py::class_<ExecuteProgramRequest>(m, "ExecuteProgramRequest",
                                    py::module_local())
      .def(py::init<Program>(), py::arg("program"),
           "Create an ExecuteProgramRequest with a program")
      .def_readonly("program", &ExecuteProgramRequest::program,
                    "The program to execute")
      .def("__str__", &ExecuteProgramRequest::ToString)
      .def("__repr__", &ExecuteProgramRequest::ToString);
}
//==============================================================================
void InitExecuteProgramResponsePybind(py::module_& m) {
  py::class_<ExecuteProgramResponse>(m, "ExecuteProgramResponse",
                                     py::module_local())
      .def(py::init<RequestId, ErrorCode>(), py::arg("request_id"),
           py::arg("error_code") = ErrorCode::kSuccess,
           "Create an ExecuteProgramResponse with request ID and error code")
      .def_readonly("request_id", &ExecuteProgramResponse::request_id,
                    "The request ID this response corresponds to")
      .def_readonly("error_code", &ExecuteProgramResponse::error_code,
                    "The error code of the response")
      .def("__str__", &ExecuteProgramResponse::ToString)
      .def("__repr__", &ExecuteProgramResponse::ToString);
}
//==============================================================================
void InitMessagesPybindSubmodule(py::module_& pm) {
  auto m = pm.def_submodule("messages", "Messages submodule");

  InitRegisterTensorShardRequestPybind(m);
  InitRegisterTensorShardResponsePybind(m);
  InitExecuteProgramRequestPybind(m);
  InitExecuteProgramResponsePybind(m);
}
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
