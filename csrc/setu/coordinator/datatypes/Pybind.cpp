//==============================================================================
// Copyright 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License")
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
#include "commons/datatypes/TensorShard.h"
#include "commons/enums/Enums.h"
#include "coordinator/datatypes/Instruction.h"
#include "coordinator/datatypes/Plan.h"
#include "coordinator/datatypes/Program.h"
#include "coordinator/datatypes/TensorMetadata.h"
#include "coordinator/datatypes/TensorOwnershipMap.h"
#include <boost/uuid/uuid_io.hpp>
//==============================================================================
namespace setu::coordinator::datatypes {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorSelectionPtr;
using setu::commons::datatypes::TensorShardsMap;
using setu::commons::enums::DType;
using setu::coordinator::datatypes::CopyInstruction;
using setu::coordinator::datatypes::InitCommInstruction;
using setu::coordinator::datatypes::Instruction;
using setu::coordinator::datatypes::Plan;
using setu::coordinator::datatypes::Program;
using setu::coordinator::datatypes::ReceiveInstruction;
using setu::coordinator::datatypes::SendInstruction;
using setu::coordinator::datatypes::TensorMetadata;
using setu::coordinator::datatypes::TensorOwnershipMap;
using setu::coordinator::datatypes::TensorOwnershipMapPtr;
using setu::coordinator::datatypes::UseCommInstruction;
//==============================================================================
void InitCopyInstructionPybind(py::module_& m) {
  py::class_<CopyInstruction>(m, "CopyInstruction")
      .def(py::init<std::pair<TensorName, ShardId>, std::size_t,
                    std::pair<TensorName, ShardId>, std::size_t, DType,
                    std::size_t>(),
           py::arg("src_tensor"), py::arg("src_memory_offset_bytes"),
           py::arg("dst_tensor"), py::arg("dst_memory_offset_bytes"),
           py::arg("dtype"), py::arg("num_elements"),
           "Create a copy instruction for GPU memory transfer")
      .def_readonly("src_tensor", &CopyInstruction::src_tensor,
                    "Source tensor (name, shard_id) pair")
      .def_readonly("src_memory_offset_bytes",
                    &CopyInstruction::src_memory_offset_bytes,
                    "Byte offset in source memory")
      .def_readonly("dst_tensor", &CopyInstruction::dst_tensor,
                    "Destination tensor (name, shard_id) pair")
      .def_readonly("dst_memory_offset_bytes",
                    &CopyInstruction::dst_memory_offset_bytes,
                    "Byte offset in destination memory")
      .def_readonly("dtype", &CopyInstruction::dtype, "Data type of elements")
      .def_readonly("num_elements", &CopyInstruction::num_elements,
                    "Number of elements to copy")
      .def("__str__", &CopyInstruction::ToString)
      .def("__repr__", &CopyInstruction::ToString);
}
//==============================================================================
void InitSendInstructionPybind(py::module_& m) {
  py::class_<SendInstruction>(m, "SendInstruction")
      .def(py::init<DeviceRank, std::pair<TensorName, ShardId>, DType,
                    std::size_t, std::size_t>(),
           py::arg("dst_device_id"), py::arg("src_tensor"), py::arg("dtype"),
           py::arg("memory_offset_bytes"), py::arg("num_elements"),
           "Create a send instruction for NCCL point-to-point communication")
      .def_readonly("dst_device_id", &SendInstruction::dst_device_id,
                    "Destination device rank")
      .def_readonly("src_tensor", &SendInstruction::src_tensor,
                    "Source tensor (name, shard_id) pair")
      .def_readonly("dtype", &SendInstruction::dtype, "Data type of elements")
      .def_readonly("memory_offset_bytes", &SendInstruction::memory_offset_bytes,
                    "Byte offset in source memory")
      .def_readonly("num_elements", &SendInstruction::num_elements,
                    "Number of elements to send")
      .def("__str__", &SendInstruction::ToString)
      .def("__repr__", &SendInstruction::ToString);
}
//==============================================================================
void InitReceiveInstructionPybind(py::module_& m) {
  py::class_<ReceiveInstruction>(m, "ReceiveInstruction")
      .def(py::init<DeviceRank, std::pair<TensorName, ShardId>, DType,
                    std::size_t, std::size_t>(),
           py::arg("src_device_id"), py::arg("dst_tensor"), py::arg("dtype"),
           py::arg("memory_offset_bytes"), py::arg("num_elements"),
           "Create a receive instruction for NCCL point-to-point communication")
      .def_readonly("src_device_id", &ReceiveInstruction::src_device_id,
                    "Source device rank")
      .def_readonly("dst_tensor", &ReceiveInstruction::dst_tensor,
                    "Destination tensor (name, shard_id) pair")
      .def_readonly("dtype", &ReceiveInstruction::dtype, "Data type of elements")
      .def_readonly("memory_offset_bytes",
                    &ReceiveInstruction::memory_offset_bytes,
                    "Byte offset in destination memory")
      .def_readonly("num_elements", &ReceiveInstruction::num_elements,
                    "Number of elements to receive")
      .def("__str__", &ReceiveInstruction::ToString)
      .def("__repr__", &ReceiveInstruction::ToString);
}
//==============================================================================
void InitInitCommInstructionPybind(py::module_& m) {
  py::class_<InitCommInstruction>(m, "InitCommInstruction")
      .def(py::init<ncclUniqueId,
                    std::unordered_map<DeviceRank, std::int32_t>>(),
           py::arg("comm_id"), py::arg("device_to_rank"),
           "Create an instruction to initialize an NCCL communicator")
      .def_readonly("comm_id", &InitCommInstruction::comm_id,
                    "NCCL unique communicator ID")
      .def_readonly("device_to_rank", &InitCommInstruction::device_to_rank,
                    "Mapping from device rank to NCCL rank")
      .def("__str__", &InitCommInstruction::ToString)
      .def("__repr__", &InitCommInstruction::ToString);
}
//==============================================================================
void InitUseCommInstructionPybind(py::module_& m) {
  py::class_<UseCommInstruction>(m, "UseCommInstruction")
      .def(py::init<ncclUniqueId>(), py::arg("comm_id"),
           "Create an instruction to switch to an existing NCCL communicator")
      .def_readonly("comm_id", &UseCommInstruction::comm_id,
                    "NCCL unique communicator ID to use")
      .def("__str__", &UseCommInstruction::ToString)
      .def("__repr__", &UseCommInstruction::ToString);
}
//==============================================================================
void InitInstructionPybind(py::module_& m) {
  py::class_<Instruction>(m, "Instruction")
      .def(py::init<CopyInstruction>(), py::arg("copy_instruction"),
           "Create instruction from CopyInstruction")
      .def(py::init<SendInstruction>(), py::arg("send_instruction"),
           "Create instruction from SendInstruction")
      .def(py::init<ReceiveInstruction>(), py::arg("receive_instruction"),
           "Create instruction from ReceiveInstruction")
      .def(py::init<InitCommInstruction>(), py::arg("init_comm_instruction"),
           "Create instruction from InitCommInstruction")
      .def(py::init<UseCommInstruction>(), py::arg("use_comm_instruction"),
           "Create instruction from UseCommInstruction")
      .def(
          "embellish",
          [](Instruction& self, py::function py_resolver) {
            self.Embellish(
                [&py_resolver](const TensorName& name, const ShardId& shard_id) {
                  py::object result = py_resolver(
                      py::cast(name), py::cast(boost::uuids::to_string(shard_id)));
                  auto ptr =
                      reinterpret_cast<DevicePtr>(result.cast<intptr_t>());
                  return ptr;
                });
          },
          py::arg("resolver"),
          "Resolve (tensor_name, shard_id) to device pointer. Resolver must "
          "return int (e.g. tensor.data_ptr()).")
      .def("__str__", &Instruction::ToString)
      .def("__repr__", &Instruction::ToString);
}
//==============================================================================
void InitProgramPybind(py::module_& m) {
  py::class_<Program>(m, "Program")
      .def(py::init<>(), "Create an empty program")
      .def_readwrite("participating_workers", &Program::participating_workers,
                     "Participating worker device ranks")
      .def_readwrite("instrs", &Program::instrs,
                     "Instructions to execute in order")
      .def("__str__", &Program::ToString)
      .def("__repr__", &Program::ToString);
}
//==============================================================================
void InitPlanPybind(py::module_& m) {
  py::class_<Plan>(m, "Plan")
      .def(py::init<>(), "Create an empty plan")
      .def_readwrite("worker_programs", &Plan::worker_programs,
                     "Mapping of device ranks to worker programs")
      .def("__str__", &Plan::ToString)
      .def("__repr__", &Plan::ToString);
}
//==============================================================================
void InitTensorMetadataPybind(py::module_& m) {
  py::class_<TensorMetadata>(m, "TensorMetadata", py::module_local())
      .def(py::init<TensorName, TensorDimMap, DType, TensorShardsMap>(),
           py::arg("name"), py::arg("dims"), py::arg("dtype"),
           py::arg("shards"))
      .def_readonly("name", &TensorMetadata::name, "Name of the tensor")
      .def_readonly("dims", &TensorMetadata::dims,
                    "Map of dimension names to TensorDim objects")
      .def_readonly("dtype", &TensorMetadata::dtype,
                    "Data type of tensor elements")
      .def_readonly("shards", &TensorMetadata::shards,
                    "Map of node IDs to tensor shards")
      .def_readonly("size", &TensorMetadata::size,
                    "Total number of elements in the tensor")
      .def("get_size", &TensorMetadata::GetSize,
           "Get total number of elements in the tensor")
      .def("get_ownership_map", &TensorMetadata::GetOwnershipMap,
           py::arg("selection"), "Get ownership map for a tensor selection")
      .def("__str__", &TensorMetadata::ToString)
      .def("__repr__", &TensorMetadata::ToString);
}
//==============================================================================
void InitTensorOwnershipMapPybind(py::module_& m) {
  py::class_<TensorOwnershipMap, TensorOwnershipMapPtr>(m, "TensorOwnershipMap",
                                                        py::module_local())
      .def(py::init<TensorSelectionPtr, TensorShardsMap>(),
           py::arg("selection"), py::arg("shards"))
      .def_readonly("shard_mapping", &TensorOwnershipMap::shard_mapping,
                    "Vector of (selection subset, owning shard) pairs")
      .def("get_num_shards", &TensorOwnershipMap::GetNumShards,
           "Get number of shard ownership mappings")
      .def("__str__", &TensorOwnershipMap::ToString)
      .def("__repr__", &TensorOwnershipMap::ToString);
}
//==============================================================================
void InitDatatypesPybindSubmodule(py::module_& pm) {
  auto m = pm.def_submodule("datatypes", "Coordinator datatypes submodule");

  // Instruction types (must be registered before Instruction itself)
  InitCopyInstructionPybind(m);
  InitSendInstructionPybind(m);
  InitReceiveInstructionPybind(m);
  InitInitCommInstructionPybind(m);
  InitUseCommInstructionPybind(m);
  InitInstructionPybind(m);

  InitProgramPybind(m);
  InitPlanPybind(m);
  InitTensorMetadataPybind(m);
  InitTensorOwnershipMapPybind(m);
}
//==============================================================================
}  // namespace setu::coordinator::datatypes
//==============================================================================
