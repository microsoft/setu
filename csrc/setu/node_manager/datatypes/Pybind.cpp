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
#include "Pybind.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
//==============================================================================
namespace setu::node_manager::datatypes {
//==============================================================================
void InitDatatypesPybindSubmodule(py::module_& pm) {
  // Node manager datatypes submodule placeholder
  // All shard types have been moved to commons/datatypes
  auto m = pm.def_submodule("datatypes", "Node manager datatypes submodule");
}
//==============================================================================
}  // namespace setu::node_manager::datatypes
//==============================================================================
