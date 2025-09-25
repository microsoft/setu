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
void InitPybindSubmodule(py::module_& pm) {
  auto m = pm.def_submodule("utils", "Utils submodule");

  // Expose the C++ clock function
  m.def("now_s", &now_s, "Get current time in seconds using C++ clock");

  InitZmqHelperPybindClass(m);
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
