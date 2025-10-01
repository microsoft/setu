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
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/Device.h"
#include "commons/datatypes/TensorDim.h"
#include "commons/datatypes/TensorDimShard.h"
#include "commons/datatypes/TensorSelection.h"
#include "commons/datatypes/TensorShard.h"
#include "commons/datatypes/TensorShardHandle.h"
#include "commons/datatypes/TensorSlice.h"
#include "commons/enums/Enums.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
using setu::commons::enums::DeviceKind;
using setu::commons::enums::DType;
//==============================================================================
void InitDevicePybind(py::module_& m) {
  py::class_<Device>(m, "Device", py::module_local())
      .def(py::init<DeviceKind, NodeRank, DeviceRank, LocalDeviceRank>(),
           py::arg("kind"), py::arg("node_rank"), py::arg("device_rank"),
           py::arg("local_device_rank"))
      .def_readonly("kind", &Device::kind, "Type of device (CPU, GPU, etc.)")
      .def_readonly("node_rank", &Device::node_rank,
                    "Rank of the node containing this device")
      .def_readonly("device_rank", &Device::device_rank,
                    "Global rank across all devices")
      .def_readonly("local_device_rank", &Device::local_device_rank,
                    "Local rank within the node")
      .def("__str__", &Device::ToString)
      .def("__repr__", &Device::ToString);
}
//==============================================================================
void InitTensorSlicePybind(py::module_& m) {
  py::class_<TensorSlice, TensorSlicePtr>(m, "TensorSlice", py::module_local())
      .def(py::init<TensorDimName, TensorIndex, TensorIndex>(),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def_readonly("dim_name", &TensorSlice::dim_name,
                    "Name of the dimension being sliced")
      .def_readonly("start", &TensorSlice::start,
                    "Starting index (inclusive) of the slice")
      .def_readonly("end", &TensorSlice::end,
                    "Ending index (exclusive) of the slice")
      .def_readonly("size", &TensorSlice::size,
                    "Size of the slice (end - start)")
      .def("__str__", &TensorSlice::ToString)
      .def("__repr__", &TensorSlice::ToString);
}
//==============================================================================
void InitTensorDimPybind(py::module_& m) {
  py::class_<TensorDim>(m, "TensorDim", py::module_local())
      .def(py::init<TensorDimName, std::size_t>(), py::arg("name"),
           py::arg("size"))
      .def_readonly("name", &TensorDim::name, "Name of the tensor dimension")
      .def_readonly("size", &TensorDim::size, "Size of the tensor dimension")
      .def("__str__", &TensorDim::ToString)
      .def("__repr__", &TensorDim::ToString);
}
//==============================================================================
//==============================================================================
void InitTensorSelectionPybind(py::module_& m) {
  py::class_<TensorSelection, TensorSelectionPtr>(m, "TensorSelection",
                                                  py::module_local())
      .def(py::init<TensorName, TensorDimMap>(), py::arg("name"),
           py::arg("dims"))
      .def(py::init<TensorName, TensorIndicesMap>(), py::arg("name"),
           py::arg("indices"))
      .def_readonly("name", &TensorSelection::name, "Name of the tensor")
      .def("get_intersection", &TensorSelection::GetIntersection,
           py::arg("other"), "Get intersection with another selection")
      .def("is_spanning", &TensorSelection::IsSpanning,
           "Check if selection spans all dimensions")
      .def("is_empty", &TensorSelection::IsEmpty, "Check if selection is empty")
      .def("is_compatible", &TensorSelection::IsCompatible, py::arg("other"),
           "Check if compatible with another selection")
      .def(
          "where",
          py::overload_cast<const TensorDimName&, const std::set<TensorIndex>&>(
              &TensorSelection::Where, py::const_),
          py::arg("dim_name"), py::arg("index_set"),
          "Create new selection with specified indices for a dimension")
      .def("where",
           py::overload_cast<const TensorDimName&, TensorSlicePtr>(
               &TensorSelection::Where, py::const_),
           py::arg("dim_name"), py::arg("slice"),
           "Create new selection with specified slice for a dimension")
      .def("__str__", &TensorSelection::ToString)
      .def("__repr__", &TensorSelection::ToString);
}
//==============================================================================
//==============================================================================
void InitCopySpecPybind(py::module_& m) {
  py::class_<CopySpec, CopySpecPtr>(m, "CopySpec", py::module_local())
      .def(py::init<TensorName, TensorName, TensorSelectionPtr,
                    TensorSelectionPtr>(),
           py::arg("src_name"), py::arg("dst_name"), py::arg("src_selection"),
           py::arg("dst_selection"))
      .def_readonly("src_name", &CopySpec::src_name,
                    "Name of the source tensor")
      .def_readonly("dst_name", &CopySpec::dst_name,
                    "Name of the destination tensor")
      .def_readonly("src_selection", &CopySpec::src_selection,
                    "Selection from the source tensor")
      .def_readonly("dst_selection", &CopySpec::dst_selection,
                    "Selection for the destination tensor")
      .def("__str__", &CopySpec::ToString)
      .def("__repr__", &CopySpec::ToString);
}
//==============================================================================
void InitTensorDimShardPybind(py::module_& m) {
  py::class_<TensorDimShard>(m, "TensorDimShard", py::module_local())
      .def(py::init<TensorDimName, ShardId, std::size_t, TensorSlicePtr,
                    std::size_t>(),
           py::arg("name"), py::arg("shard_id"), py::arg("dim_size"),
           py::arg("slice"), py::arg("stride"))
      .def_readonly("name", &TensorDimShard::name,
                    "Name of the tensor dimension")
      .def_readonly("shard_id", &TensorDimShard::shard_id,
                    "Unique identifier for this shard")
      .def_readonly("dim_size", &TensorDimShard::dim_size,
                    "Total size of the original dimension")
      .def_readonly("shard_size", &TensorDimShard::shard_size,
                    "Size of this specific shard")
      .def_readonly("slice", &TensorDimShard::slice,
                    "Slice specification for this shard")
      .def_readonly("stride", &TensorDimShard::stride,
                    "Memory stride for accessing elements")
      .def("__str__", &TensorDimShard::ToString)
      .def("__repr__", &TensorDimShard::ToString);
}
//==============================================================================
void InitTensorShardPybind(py::module_& m) {
  py::class_<TensorShard, TensorShardPtr>(m, "TensorShard", py::module_local())
      .def(py::init<TensorName, ShardId, Device, DevicePtr, DType,
                    TensorDimShardsMap>(),
           py::arg("name"), py::arg("id"), py::arg("device"),
           py::arg("device_ptr"), py::arg("dtype"), py::arg("dim_shards"))
      .def_readonly("id", &TensorShard::id, "Unique identifier for this shard")
      .def_readonly("name", &TensorShard::name,
                    "Name of the tensor being sharded")
      .def_readonly("device", &TensorShard::device,
                    "Device where this shard resides")
      .def_readonly("device_ptr", &TensorShard::device_ptr,
                    "Pointer to device memory location")
      .def_readonly("dtype", &TensorShard::dtype,
                    "Data type of tensor elements")
      .def_readonly("dim_shards", &TensorShard::dim_shards,
                    "Map of dimension names to shard info")
      .def_readonly("shard_size", &TensorShard::shard_size,
                    "Size of this specific shard")
      .def("get_shard_size", &TensorShard::GetShardSize,
           "Get total number of elements in the shard")
      .def("get_num_dims", &TensorShard::GetNumDims,
           "Get number of dimensions in this shard")
      .def("get_dim_slice", &TensorShard::GetDimSlice, py::arg("dim_name"),
           "Get slice information for a specific dimension")
      .def("__str__", &TensorShard::ToString)
      .def("__repr__", &TensorShard::ToString);
}
//==============================================================================
void InitTensorShardReadHandlePybind(py::module_& m) {
  py::class_<TensorShardReadHandle, TensorShardReadHandlePtr>(
      m, "TensorShardReadHandle", py::module_local())
      .def(py::init<TensorShardPtr>(), py::arg("shard"),
           "Create read handle and acquire shared lock")
      .def("get_device_ptr", &TensorShardReadHandle::GetDevicePtr,
           "Get read-only pointer to device memory")
      .def("get_shard", &TensorShardReadHandle::GetShard,
           py::return_value_policy::reference,
           "Get the tensor shard being accessed");
}
//==============================================================================
void InitTensorShardWriteHandlePybind(py::module_& m) {
  py::class_<TensorShardWriteHandle, TensorShardWriteHandlePtr>(
      m, "TensorShardWriteHandle", py::module_local())
      .def(py::init<TensorShardPtr>(), py::arg("shard"),
           "Create write handle and acquire exclusive lock")
      .def("get_device_ptr", &TensorShardWriteHandle::GetDevicePtr,
           "Get read-write pointer to device memory")
      .def("get_shard", &TensorShardWriteHandle::GetShard,
           py::return_value_policy::reference,
           "Get the tensor shard being accessed");
}
//==============================================================================
void InitDatatypesPybindSubmodule(py::module_& pm) {
  auto m = pm.def_submodule("datatypes", "Datatypes submodule");

  InitDevicePybind(m);
  InitTensorSlicePybind(m);
  InitTensorDimPybind(m);
  InitTensorSelectionPybind(m);
  InitCopySpecPybind(m);
  InitTensorDimShardPybind(m);
  InitTensorShardPybind(m);
  InitTensorShardReadHandlePybind(m);
  InitTensorShardWriteHandlePybind(m);
}
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
