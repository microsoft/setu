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
#include "commons/Time.h"
#include "commons/utils/ZmqHelper.h"
#include "commons/utils/TorchTensorIPC.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::time_utils::now_s;
//==============================================================================
void InitZmqHelperPybindClass(py::module_& m) {
  // We need to create specific instantiations for the Python bindings
  // since ZmqHelper uses templates
  py::class_<zmq::socket_t, ZmqSocketPtr>(m, "ZmqSocket", py::module_local())
      .def(py::init<zmq::context_t&, int>(), py::arg("context"),
           py::arg("type"))
      .def("bind",
           static_cast<void (zmq::socket_t::*)(const std::string&)>(
               &zmq::socket_t::bind),
           py::arg("endpoint"), "Bind the socket to an endpoint")
      .def("connect",
           static_cast<void (zmq::socket_t::*)(const std::string&)>(
               &zmq::socket_t::connect),
           py::arg("endpoint"), "Connect the socket to an endpoint")
      .def(
          "setsockopt_string",
          [](zmq::socket_t& socket, int option, const std::string& value) {
// Revert to using setsockopt with a compiler warning suppression
// This is a temporary solution until we can properly update to the newer ZMQ
// API
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            socket.setsockopt(option, value.c_str(), value.size());
#pragma GCC diagnostic pop
          },
          py::arg("option"), py::arg("value"),
          "Set a socket option with a string value");

  py::class_<zmq::context_t, ZmqContextPtr>(m, "ZmqContext", py::module_local())
      .def(py::init<>());
}
//==============================================================================
void InitTorchTensorIPCPybindClass(py::module_& m) {
    py::class_<TensorIPCSpec, TensorIPCSpecPtr>(m, "TensorIPCSpec", py::module_local())
        .def(py::init<torch::IntArrayRef, torch::IntArrayRef, std::int64_t,
                      torch::Dtype, bool, std::int32_t, std::string,
                      std::uint64_t, std::uint64_t, std::string, std::uint64_t,
                      cudaIpcEventHandle_t, bool>(),
             py::arg("tensor_size"),
             py::arg("tensor_stride"),
             py::arg("tensor_offset"),
             py::arg("dtype"),
             py::arg("requires_grad"),
             py::arg("storage_device"),
             py::arg("storage_handle"),
             py::arg("storage_size_bytes"),
             py::arg("storage_offset_bytes"),
             py::arg("ref_counter_handle"),
             py::arg("ref_counter_offset"),
             py::arg("event_handle"),
             py::arg("event_sync_required"))
        .def_readonly("tensor_size", &TensorIPCSpec::tensor_size,
                     "Tensor size (shape)")
        .def_readonly("tensor_stride", &TensorIPCSpec::tensor_stride,
                     "Tensor stride")
        .def_readonly("tensor_offset", &TensorIPCSpec::tensor_offset,
                     "Tensor storage offset")
        .def_readonly("dtype", &TensorIPCSpec::dtype,
                     "Tensor data type")
        .def_readonly("requires_grad", &TensorIPCSpec::requires_grad,
                     "Whether tensor requires gradient")
        .def_readonly("storage_device", &TensorIPCSpec::storage_device,
                     "Storage device (e.g., cuda:0)")
        .def_property_readonly("storage_handle",
                              [](const TensorIPCSpec& spec) {
                                  return py::bytes(spec.storage_handle);
                              },
                              "CUDA IPC memory handle (as bytes)")
        .def_readonly("storage_size_bytes", &TensorIPCSpec::storage_size_bytes, "Storage size in bytes")
        .def_readonly("storage_offset_bytes", &TensorIPCSpec::storage_offset_bytes,
                     "Storage offset in bytes")
        .def_property_readonly("ref_counter_handle",
                              [](const TensorIPCSpec& spec) {
                                  return py::bytes(spec.ref_counter_handle);
                              },
                              "Reference counter IPC handle (as bytes)")
        .def_readonly("ref_counter_offset", &TensorIPCSpec::ref_counter_offset,
                     "Reference counter offset")
        .def_property_readonly("event_handle",
                              [](const TensorIPCSpec& spec) {
                                  return py::bytes(
                                      reinterpret_cast<const char*>(&spec.event_handle),
                                      CUDA_IPC_HANDLE_SIZE);
                              },
                              "CUDA IPC event handle (as bytes)")
        .def_readonly("event_sync_required", &TensorIPCSpec::event_sync_required,
                     "Whether event synchronization is required")
        .def("to_dict",
             [](const TensorIPCSpec& spec) {
                 py::dict d;
                 d["tensor_size"] = spec.tensor_size;
                 d["tensor_stride"] = spec.tensor_stride;
                 d["tensor_offset"] = spec.tensor_offset;
                 d["dtype"] = spec.dtype;
                 d["requires_grad"] = spec.requires_grad;
                 d["storage_device"] = spec.storage_device;
                 d["storage_handle"] = py::bytes(spec.storage_handle);
                 d["storage_size_bytes"] = spec.storage_size_bytes;
                 d["storage_offset_bytes"] = spec.storage_offset_bytes;
                 d["ref_counter_handle"] = py::bytes(spec.ref_counter_handle);
                 d["ref_counter_offset"] = spec.ref_counter_offset;
                 d["event_handle"] = py::bytes(
                     reinterpret_cast<const char*>(&spec.event_handle),
                     CUDA_IPC_HANDLE_SIZE);
                 d["event_sync_required"] = spec.event_sync_required;
                 return d;
             },
             "Convert TensorIPCSpec to a dictionary for splatting into function arguments");

    m.def("prepare_tensor_ipc_spec", &PrepareTensorIPCSpec,
          py::arg("tensor"),
          "Prepare a tensor for IPC sharing. Returns a TensorIPCSpec containing "
          "all necessary information to reconstruct the tensor in another process.");
}
//==============================================================================
void InitPybindSubmodule(py::module_& pm) {
  auto m = pm.def_submodule("utils", "Utils submodule");

  // Expose the C++ clock function
  m.def("now_s", &now_s, "Get current time in seconds using C++ clock");

  InitZmqHelperPybindClass(m);
  InitTorchTensorIPCPybindClass(m);
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
