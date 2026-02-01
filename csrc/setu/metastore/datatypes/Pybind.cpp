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
#include "commons/TorchCommon.h"
#include "commons/datatypes/TensorDim.h"
#include "commons/datatypes/TensorSelection.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "metastore/datatypes/TensorMetadata.h"
#include "metastore/datatypes/TensorOwnershipMap.h"
//==============================================================================
namespace setu::metastore::datatypes {
//==============================================================================
using setu::commons::NodeId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorSelectionPtr;
using setu::commons::datatypes::TensorShardSpecPtr;
using TensorShardSpecMap = std::unordered_map<ShardId, TensorShardSpecPtr>;
using ShardOwnerMap = std::unordered_map<ShardId, NodeId>;
//==============================================================================
void InitTensorMetadataPybind(py::module_& m) {
  py::class_<TensorMetadata>(m, "TensorMetadata", py::module_local())
      .def(py::init<TensorName, TensorDimMap, torch::Dtype, TensorShardSpecMap,
                    ShardOwnerMap>(),
           py::arg("name"), py::arg("dims"), py::arg("dtype"),
           py::arg("shards"), py::arg("shard_owners"))
      .def_readonly("name", &TensorMetadata::name, "Name of the tensor")
      .def_readonly("dims", &TensorMetadata::dims,
                    "Map of dimension names to TensorDim objects")
      .def_readonly("dtype", &TensorMetadata::dtype,
                    "Data type of tensor elements")
      .def_readonly("shards", &TensorMetadata::shards,
                    "Map of shard IDs to tensor shard specs")
      .def_readonly("shard_owners", &TensorMetadata::shard_owners,
                    "Map of shard IDs to owning NodeIds")
      .def_readonly("size", &TensorMetadata::size,
                    "Total number of elements in the tensor")
      .def("get_size", &TensorMetadata::GetSize,
           "Get total number of elements in the tensor")
      .def("get_owner_node_ids", &TensorMetadata::GetOwnerNodeIds,
           "Get all unique NodeIds that own shards of this tensor")
      .def("get_ownership_map", &TensorMetadata::GetOwnershipMap,
           py::arg("selection"), "Get ownership map for a tensor selection")
      .def("__str__", &TensorMetadata::ToString)
      .def("__repr__", &TensorMetadata::ToString);
}
//==============================================================================
void InitTensorOwnershipMapPybind(py::module_& m) {
  py::class_<TensorOwnershipMap, TensorOwnershipMapPtr>(m, "TensorOwnershipMap",
                                                        py::module_local())
      .def(py::init<TensorSelectionPtr, const TensorShardSpecMap&>(),
           py::arg("selection"), py::arg("shards"))
      .def_readonly("shard_mapping", &TensorOwnershipMap::shard_mapping,
                    "Vector of (selection subset, owning shard spec) pairs")
      .def("get_num_shards", &TensorOwnershipMap::GetNumShards,
           "Get number of shard ownership mappings")
      .def("__str__", &TensorOwnershipMap::ToString)
      .def("__repr__", &TensorOwnershipMap::ToString);
}
//==============================================================================
void InitDatatypesPybindSubmodule(py::module_& pm) {
  auto m = pm.def_submodule("datatypes", "Coordinator datatypes submodule");

  InitTensorMetadataPybind(m);
  InitTensorOwnershipMapPybind(m);
}
//==============================================================================
}  // namespace setu::metastore::datatypes
//==============================================================================
