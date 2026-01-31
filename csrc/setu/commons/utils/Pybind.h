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
#pragma once
//==============================================================================
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "commons/TorchCommon.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
void InitPybindSubmodule(py::module_& pm);
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
// Pybind11 type caster for boost::uuids::uuid
//==============================================================================
namespace pybind11::detail {
template <>
struct type_caster<boost::uuids::uuid> {
 public:
  PYBIND11_TYPE_CASTER(boost::uuids::uuid, const_name("uuid.UUID"));

  // Python -> C++ conversion
  bool load(handle src, bool) {
    // Import Python's uuid module
    auto uuid_module = module_::import("uuid");
    auto uuid_class = uuid_module.attr("UUID");

    // Check if it's a UUID instance
    if (!isinstance(src, uuid_class)) {
      return false;
    }

    // Get the bytes representation from Python UUID
    auto bytes_obj = src.attr("bytes");
    auto bytes_str = bytes_obj.cast<std::string>();

    // Ensure we have exactly 16 bytes
    if (bytes_str.size() != 16) {
      return false;
    }

    // Copy bytes into boost::uuids::uuid
    std::copy(bytes_str.begin(), bytes_str.end(), value.begin());
    return true;
  }

  // C++ -> Python conversion
  static handle cast(const boost::uuids::uuid& src, return_value_policy,
                     handle) {
    // Import Python's uuid module
    auto uuid_module = module_::import("uuid");
    auto uuid_class = uuid_module.attr("UUID");

    // Convert boost UUID to bytes
    std::string bytes_str(src.begin(), src.end());
    auto bytes_obj = pybind11::bytes(bytes_str);

    // Create Python UUID from bytes
    return uuid_class(arg("bytes") = bytes_obj).release();
  }
};
}  // namespace pybind11::detail
//==============================================================================
