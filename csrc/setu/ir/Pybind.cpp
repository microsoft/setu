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
#include "setu/ir/Pybind.h"
//==============================================================================
#include <boost/uuid/uuid_io.hpp>
#include <nccl.h>
//==============================================================================
#include "setu/commons/Logging.h"
#include "setu/commons/StdCommon.h"
#include "setu/commons/TorchCommon.h"
#include "setu/commons/datatypes/TensorShardIdentifier.h"
#include "setu/ir/Instruction.h"
//==============================================================================
namespace setu::ir {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::datatypes::TensorShardIdentifier;
//==============================================================================
void InitCopyInstructionPybind(py::module_& m) {
  py::class_<CopyInstruction>(m, "CopyInstruction")
      .def(py::init<TensorShardIdentifier, std::size_t, TensorShardIdentifier,
                    std::size_t, torch::Dtype, std::size_t>(),
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
      .def(py::init<DeviceRank, TensorShardIdentifier, torch::Dtype,
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
      .def(py::init<DeviceRank, TensorShardIdentifier, torch::Dtype,
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
            self.Embellish([&py_resolver](const TensorShardIdentifier& id) {
              py::object result = py_resolver(
                  py::cast(id.tensor_name),
                  py::cast(boost::uuids::to_string(id.shard_id)));
              auto ptr = reinterpret_cast<DevicePtr>(result.cast<intptr_t>());
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
ncclUniqueId GenerateNcclUniqueId() {
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  return id;
}
//==============================================================================
void InitNcclUniqueIdPybind(py::module_& m) {
  // Register ncclUniqueId as an opaque type that can be passed around
  py::class_<ncclUniqueId>(m, "NcclUniqueId")
      .def(py::init<>())
      .def("__repr__", [](const ncclUniqueId& id) {
        // Show first few bytes as hex for debugging
        std::string hex;
        for (std::size_t i = 0; i < 8 && i < NCCL_UNIQUE_ID_BYTES; ++i) {
          hex += std::format("{:02x}", static_cast<unsigned char>(id.internal[i]));
        }
        return std::format("NcclUniqueId({}...)", hex);
      });
}
//==============================================================================
void InitIrPybind(py::module_& m) {
  // Register ncclUniqueId type first (needed by InitCommInstruction)
  InitNcclUniqueIdPybind(m);

  // Utility function to generate NCCL unique IDs
  m.def("generate_nccl_id", &GenerateNcclUniqueId,
        "Generate a new NCCL unique ID for communicator initialization");

  // Instruction types (must be registered before Instruction itself)
  InitCopyInstructionPybind(m);
  InitSendInstructionPybind(m);
  InitReceiveInstructionPybind(m);
  InitInitCommInstructionPybind(m);
  InitUseCommInstructionPybind(m);
  InitInstructionPybind(m);
}
//==============================================================================
}  // namespace setu::ir
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  setu::commons::Logger::InitializeLogLevel();
  setu::ir::InitIrPybind(m);
}
//==============================================================================