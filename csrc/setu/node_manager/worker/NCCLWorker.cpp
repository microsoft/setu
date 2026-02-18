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
#include "node_manager/worker/NCCLWorker.h"

#include "node_manager/worker/Worker.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/Comm.h"
#include "commons/utils/ThreadingUtils.h"
#include "messaging/Messages.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::enums::ErrorCode;
using setu::commons::messages::ExecuteProgramRequest;
using setu::commons::messages::ExecuteProgramResponse;
using setu::commons::utils::Comm;
using setu::commons::utils::ZmqHelper;
using setu::planner::Participant;
//==============================================================================

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = (call);                                           \
    ASSERT_VALID_RUNTIME(err == cudaSuccess, "CUDA error: {} at {}:{}", \
                         cudaGetErrorString(err), __FILE__, __LINE__);  \
  } while (0)

#define NCCL_CHECK(call)                                                \
  do {                                                                  \
    ncclResult_t res = (call);                                          \
    ASSERT_VALID_RUNTIME(res == ncclSuccess, "NCCL error: {} at {}:{}", \
                         ncclGetErrorString(res), __FILE__, __LINE__);  \
  } while (0)

//==============================================================================
// NCCLWorker
//==============================================================================

NCCLWorker::NCCLWorker(NodeId node_id, Device device, RegisterSet register_set)
    : Worker(node_id, device),
      stream_(nullptr),
      register_file_(std::move(register_set)) {}

NCCLWorker::~NCCLWorker() {
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
  for (auto& [key, entry] : comm_cache_) {
    ncclCommDestroy(entry.nccl_comm);
  }
}

void NCCLWorker::Setup() {
  CUDA_CHECK(cudaSetDevice(device_.LocalDeviceIndex()));
  CUDA_CHECK(cudaStreamCreate(&stream_));

  if (!register_file_.Empty()) {
    register_file_.Allocate();
    LOG_DEBUG("Allocated {} registers on device {}",
              register_file_.NumRegisters(), device_);
  }

  LOG_DEBUG("NCCLWorker setup complete for device {}", device_);
}

void NCCLWorker::Execute(const Program& program) {
  bool group_started = false;

  for (const auto& instruction : program) {
    ExecuteInstruction(instruction, group_started);
  }

  if (group_started) {
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }
}

void NCCLWorker::ExecuteInstruction(const Instruction& instruction,
                                    bool& group_started) {
  std::visit(
      [this, &group_started](const auto& inst) {
        using T = std::decay_t<decltype(inst)>;

        if constexpr (std::is_same_v<T, InitComm>) {
          ExecuteInitComm(inst);
        } else if constexpr (std::is_same_v<T, UseComm>) {
          ExecuteUseComm(inst);
        } else if constexpr (std::is_same_v<T, Copy>) {
          if (!group_started) {
            NCCL_CHECK(ncclGroupStart());
            group_started = true;
          }
          ExecuteCopy(inst);
        } else if constexpr (std::is_same_v<T, Send>) {
          if (!group_started) {
            NCCL_CHECK(ncclGroupStart());
            group_started = true;
          }
          ExecuteSend(inst);
        } else if constexpr (std::is_same_v<T, Receive>) {
          if (!group_started) {
            NCCL_CHECK(ncclGroupStart());
            group_started = true;
          }
          ExecuteReceive(inst);
        }
      },
      instruction.instr);
}

//==============================================================================
// Instruction Handlers
//==============================================================================

void NCCLWorker::ExecuteInitComm(const InitComm& inst) {
  std::string key = CommIdToString(inst.comm_id);

  const std::int32_t num_ranks =
      static_cast<std::int32_t>(inst.participant_to_rank.size());
  auto part = Participant(node_id_, device_);
  const std::int32_t rank = inst.participant_to_rank.at(part);

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, inst.comm_id, rank));

  comm_cache_[key] = CommCacheEntry{.nccl_comm = comm};

  active_comm_key_ = key;
  LOG_DEBUG("InitComm complete: {} ranks, this rank={}", num_ranks, rank);
}

void NCCLWorker::ExecuteUseComm(const UseComm& inst) {
  active_comm_key_ = CommIdToString(inst.comm_id);
  LOG_DEBUG("UseComm: switched to communicator");
}

void NCCLWorker::ExecuteCopy(const Copy& inst) {
  const std::size_t bytes = inst.count * GetDTypeSizeBytes(inst.dtype);

  CUDA_CHECK(
      cudaMemcpyAsync(static_cast<char*>(inst.dst_ptr) + inst.dst_offset_bytes,
                      static_cast<char*>(inst.src_ptr) + inst.src_offset_bytes,
                      bytes, cudaMemcpyDeviceToDevice, stream_));

  LOG_DEBUG("Copy: {} bytes from {} to {}", bytes, inst.src_ref.ToString(),
            inst.dst_ref.ToString());
}

void NCCLWorker::ExecuteSend(const Send& inst) {
  auto& entry = comm_cache_.at(active_comm_key_);

  NCCL_CHECK(ncclSend(static_cast<char*>(inst.src_ptr) + inst.offset_bytes,
                      inst.count, ToNcclDataType(inst.dtype), inst.peer_rank,
                      entry.nccl_comm, stream_));

  LOG_DEBUG("Send: {} elements from {} to device rank: {}", inst.count,
            inst.src_ref.ToString(), inst.peer_rank);
}

void NCCLWorker::ExecuteReceive(const Receive& inst) {
  auto& entry = comm_cache_.at(active_comm_key_);

  NCCL_CHECK(ncclRecv(static_cast<char*>(inst.dst_ptr) + inst.offset_bytes,
                      inst.count, ToNcclDataType(inst.dtype), inst.peer_rank,
                      entry.nccl_comm, stream_));

  LOG_DEBUG("Receive: {} elements to {} from device rank: {}", inst.count,
            inst.dst_ref.ToString(), inst.peer_rank);
}

DevicePtr NCCLWorker::ResolveRegister(const RegisterRef& ref) const {
  return register_file_.GetPtr(ref.register_index);
}

//==============================================================================
// Helper Functions
//==============================================================================

std::string NCCLWorker::CommIdToString(const ncclUniqueId& id) {
  return std::string(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}

ncclDataType_t NCCLWorker::ToNcclDataType(torch::Dtype dtype) {
  switch (dtype) {
    case torch::Dtype::Float:
      return ncclFloat;
    case torch::Dtype::BFloat16:
      return ncclBfloat16;
    default:
      RAISE_RUNTIME_ERROR("Unsupported dtype: {}", static_cast<int>(dtype));
  }
}

std::size_t NCCLWorker::GetDTypeSizeBytes(torch::Dtype dtype) {
  switch (dtype) {
    case torch::Dtype::Float:
      return 4;
    case torch::Dtype::BFloat16:
      return 2;
    default:
      RAISE_RUNTIME_ERROR("Unsupported dtype: {}", static_cast<int>(dtype));
  }
}

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
