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
void InitCopyEntryPybind(py::module_& m) {
  py::class_<CopyEntry>(m, "CopyEntry")
      .def(py::init<BufferRef, std::size_t, BufferRef, std::size_t, std::size_t,
                    torch::Dtype>(),
           py::arg("src_ref"), py::arg("src_offset_bytes"), py::arg("dst_ref"),
           py::arg("dst_offset_bytes"), py::arg("count"), py::arg("dtype"),
           "Create a single copy entry for a batched Copy instruction")
      .def_readonly("src_ref", &CopyEntry::src_ref, "Source buffer reference")
      .def_readonly("src_offset_bytes", &CopyEntry::src_offset_bytes,
                    "Byte offset in source memory")
      .def_readonly("dst_ref", &CopyEntry::dst_ref,
                    "Destination buffer reference")
      .def_readonly("dst_offset_bytes", &CopyEntry::dst_offset_bytes,
                    "Byte offset in destination memory")
      .def_readonly("count", &CopyEntry::count, "Number of elements to copy")
      .def_readonly("dtype", &CopyEntry::dtype, "Data type of elements");
}
//==============================================================================
void InitCopyInstructionPybind(py::module_& m) {
  py::class_<Copy>(m, "Copy")
      .def(py::init<std::vector<CopyEntry>>(), py::arg("entries"),
           "Create a batched copy instruction from a list of CopyEntry items")
      .def(py::init<BufferRef, std::size_t, BufferRef, std::size_t, std::size_t,
                    torch::Dtype>(),
           py::arg("src_ref"), py::arg("src_offset_bytes"), py::arg("dst_ref"),
           py::arg("dst_offset_bytes"), py::arg("count"), py::arg("dtype"),
           "Create a single-entry copy instruction (convenience)")
      .def_readonly("entries", &Copy::entries, "Batched copy entries")
      .def("__str__", &Copy::ToString)
      .def("__repr__", &Copy::ToString);
}
//==============================================================================
void InitSendInstructionPybind(py::module_& m) {
  py::class_<Send>(m, "Send")
      .def(py::init<CommId, BufferRef, std::size_t, std::size_t, torch::Dtype,
                    DeviceRank>(),
           py::arg("comm_id"), py::arg("src_ref"), py::arg("offset"),
           py::arg("count"), py::arg("dtype"), py::arg("peer_rank"),
           "Create a send instruction for point-to-point communication")
      .def_readonly("comm_id", &Send::comm_id, "Communicator identifier")
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
      .def(py::init<CommId, BufferRef, std::size_t, std::size_t, torch::Dtype,
                    DeviceRank>(),
           py::arg("comm_id"), py::arg("dst_ref"), py::arg("offset_bytes"),
           py::arg("count"), py::arg("dtype"), py::arg("peer_rank"),
           "Create a receive instruction for point-to-point communication")
      .def_readonly("comm_id", &Receive::comm_id, "Communicator identifier")
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
      .def(py::init<CommId, std::unordered_map<Participant, DeviceRank>>(),
           py::arg("comm_id"), py::arg("participant_to_rank"),
           "Create an instruction to initialize a communicator")
      .def_readonly("comm_id", &InitComm::comm_id, "Communicator identifier")
      .def_readonly("participant_to_rank", &InitComm::participant_to_rank,
                    "Mapping from participant to rank")
      .def("__str__", &InitComm::ToString)
      .def("__repr__", &InitComm::ToString);
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
CommId GenerateCommId() {
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  return CommId::From(id);
}
//==============================================================================
void InitCommIdPybind(py::module_& m) {
  py::class_<CommId>(m, "CommId")
      .def(py::init<>())
      .def("__str__", &CommId::ToString)
      .def("__repr__", &CommId::ToString)
      .def("__eq__", &CommId::operator==);
}
//==============================================================================
void InitLLCPybind(py::module_& m) {
  // Register CommId type first (needed by instruction types)
  InitCommIdPybind(m);

  // Register ShardRef type (needed by instruction types)
  InitShardRefPybind(m);

  // Utility function to generate communicator IDs (backed by NCCL)
  m.def("generate_comm_id", &GenerateCommId,
        "Generate a new communicator ID for communicator initialization");

  // Instruction types (must be registered before Instruction itself)
  InitCopyEntryPybind(m);
  InitCopyInstructionPybind(m);
  InitSendInstructionPybind(m);
  InitReceiveInstructionPybind(m);
  InitInitCommInstructionPybind(m);
  InitBarrierInstructionPybind(m);
  InitInstructionPybind(m);
}
//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
