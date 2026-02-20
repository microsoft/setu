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
#include "planner/ir/llc/Pybind.h"
//==============================================================================
#include <nccl.h>

#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/uuid_io.hpp>
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/Participant.h"
#include "planner/ir/llc/Instruction.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::planner::Participant;
//==============================================================================
void InitShardRefPybind(py::module_& m) {
  py::class_<ShardRef>(m, "ShardRef")
      .def(py::init<ShardId, std::optional<TensorName>>(), py::arg("shard_id"),
           py::arg("tensor_name") = std::nullopt,
           "Create a shard reference with UUID and optional tensor name")
      .def(py::init([](const std::string& shard_id_str,
                       std::optional<TensorName> tensor_name) {
             boost::uuids::string_generator gen;
             return ShardRef(gen(shard_id_str), std::move(tensor_name));
           }),
           py::arg("shard_id_str"), py::arg("tensor_name") = std::nullopt,
           "Create a shard reference from UUID string and optional tensor name")
      .def_readonly("shard_id", &ShardRef::shard_id,
                    "Unique UUID for the shard")
      .def_readonly("node_id", &ShardRef::node_id,
                    "Node where shard resides (debug)")
      .def_readonly("tensor_name", &ShardRef::tensor_name,
                    "Parent tensor name (debug)")
      .def("__str__", &ShardRef::ToString)
      .def("__repr__", &ShardRef::ToString)
      .def("__eq__", &ShardRef::operator==);
}
//==============================================================================
void InitCopyInstructionPybind(py::module_& m) {
  py::class_<Copy>(m, "Copy")
      .def(py::init<BufferRef, std::size_t, BufferRef, std::size_t, std::size_t,
                    torch::Dtype>(),
           py::arg("src_ref"), py::arg("src_offset_bytes"), py::arg("dst_ref"),
           py::arg("dst_offset_bytes"), py::arg("count"), py::arg("dtype"),
           "Create a copy instruction for GPU memory transfer")
      .def_readonly("src_ref", &Copy::src_ref, "Source buffer reference")
      .def_readonly("src_offset_bytes", &Copy::src_offset_bytes,
                    "Byte offset in source memory")
      .def_readonly("dst_ref", &Copy::dst_ref, "Destination buffer reference")
      .def_readonly("dst_offset_bytes", &Copy::dst_offset_bytes,
                    "Byte offset in destination memory")
      .def_readonly("count", &Copy::count, "Number of elements to copy")
      .def_readonly("dtype", &Copy::dtype, "Data type of elements")
      .def("__str__", &Copy::ToString)
      .def("__repr__", &Copy::ToString);
}
//==============================================================================
void InitSendInstructionPybind(py::module_& m) {
  py::class_<Send>(m, "Send")
      .def(py::init<BufferRef, std::size_t, std::size_t, torch::Dtype,
                    DeviceRank>(),
           py::arg("src_ref"), py::arg("offset"), py::arg("count"),
           py::arg("dtype"), py::arg("peer_rank"),
           "Create a send instruction for NCCL point-to-point communication")
      .def_readonly("peer_rank", &Send::peer_rank,
                    "Destination device rank in the communicator")
      .def_readonly("src_ref", &Send::src_ref, "Source buffer reference")
      .def_readonly("offset_bytes", &Send::offset_bytes,
                    "Byte offset in source memory")
      .def_readonly("count", &Send::count, "Number of elements to send")
      .def_readonly("dtype", &Send::dtype, "Data type of elements")
      .def("__str__", &Send::ToString)
      .def("__repr__", &Send::ToString);
}
//==============================================================================
void InitReceiveInstructionPybind(py::module_& m) {
  py::class_<Receive>(m, "Receive")
      .def(py::init<BufferRef, std::size_t, std::size_t, torch::Dtype,
                    DeviceRank>(),
           py::arg("dst_ref"), py::arg("offset_bytes"), py::arg("count"),
           py::arg("dtype"), py::arg("peer_rank"),
           "Create a receive instruction for NCCL point-to-point communication")
      .def_readonly("peer_rank", &Receive::peer_rank,
                    "Source device rank in the communicator")
      .def_readonly("dst_ref", &Receive::dst_ref,
                    "Destination buffer reference")
      .def_readonly("offset_bytes", &Receive::offset_bytes,
                    "Byte offset in destination buffer")
      .def_readonly("count", &Receive::count, "Number of elements to receive")
      .def_readonly("dtype", &Receive::dtype, "Data type of elements")
      .def("__str__", &Receive::ToString)
      .def("__repr__", &Receive::ToString);
}
//==============================================================================
void InitInitCommInstructionPybind(py::module_& m) {
  py::class_<InitComm>(m, "InitComm")
      .def(
          py::init<ncclUniqueId, std::unordered_map<Participant, DeviceRank>>(),
          py::arg("comm_id"), py::arg("participant_to_rank"),
          "Create an instruction to initialize an NCCL communicator")
      .def_readonly("comm_id", &InitComm::comm_id,
                    "NCCL unique communicator ID")
      .def_readonly("participant_to_rank", &InitComm::participant_to_rank,
                    "Mapping from participant to NCCL rank")
      .def("__str__", &InitComm::ToString)
      .def("__repr__", &InitComm::ToString);
}
//==============================================================================
void InitUseCommInstructionPybind(py::module_& m) {
  py::class_<UseComm>(m, "UseComm")
      .def(py::init<ncclUniqueId>(), py::arg("comm_id"),
           "Create an instruction to switch to an existing NCCL communicator")
      .def_readonly("comm_id", &UseComm::comm_id,
                    "NCCL unique communicator ID to use")
      .def("__str__", &UseComm::ToString)
      .def("__repr__", &UseComm::ToString);
}
//==============================================================================
void InitBarrierInstructionPybind(py::module_& m) {
  py::class_<Barrier>(m, "Barrier")
      .def(py::init<>(), "Create a synchronization barrier instruction")
      .def("__str__", &Barrier::ToString)
      .def("__repr__", &Barrier::ToString);
}
//==============================================================================
void InitInstructionPybind(py::module_& m) {
  py::class_<Instruction>(m, "Instruction")
      .def(py::init<Copy>(), py::arg("copy"), "Create instruction from Copy")
      .def(py::init<Send>(), py::arg("send"), "Create instruction from Send")
      .def(py::init<Receive>(), py::arg("receive"),
           "Create instruction from Receive")
      .def(py::init<InitComm>(), py::arg("init_comm"),
           "Create instruction from InitComm")
      .def(py::init<UseComm>(), py::arg("use_comm"),
           "Create instruction from UseComm")
      .def(py::init<Barrier>(), py::arg("barrier"),
           "Create instruction from Barrier")
      .def(
          "embellish",
          [](Instruction& self, py::function py_resolver) {
            self.Embellish([&py_resolver](const BufferRef& ref) {
              ASSERT_VALID_RUNTIME(ref.IsShard(),
                                   "Python embellish only supports ShardRef");
              const auto& shard = ref.AsShard();
              py::object result =
                  py_resolver(py::cast(boost::uuids::to_string(shard.shard_id)),
                              py::cast(shard.tensor_name));
              auto ptr = reinterpret_cast<DevicePtr>(result.cast<intptr_t>());
              return ptr;
            });
          },
          py::arg("resolver"),
          "Resolve (shard_id, tensor_name) to device pointer. Resolver must "
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
          hex +=
              std::format("{:02x}", static_cast<unsigned char>(id.internal[i]));
        }
        return std::format("NcclUniqueId({}...)", hex);
      });
}
//==============================================================================
void InitLLCPybind(py::module_& m) {
  // Register ncclUniqueId type first (needed by InitCommInstruction)
  InitNcclUniqueIdPybind(m);

  // Register ShardRef type (needed by instruction types)
  InitShardRefPybind(m);

  // Utility function to generate NCCL unique IDs
  m.def("generate_nccl_id", &GenerateNcclUniqueId,
        "Generate a new NCCL unique ID for communicator initialization");

  // Instruction types (must be registered before Instruction itself)
  InitCopyInstructionPybind(m);
  InitSendInstructionPybind(m);
  InitReceiveInstructionPybind(m);
  InitInitCommInstructionPybind(m);
  InitUseCommInstructionPybind(m);
  InitBarrierInstructionPybind(m);
  InitInstructionPybind(m);
}
//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
