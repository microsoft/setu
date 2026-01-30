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
#include "node_manager/worker/Worker.h"
#include "node_manager/worker/NCCLWorker.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/messages/Messages.h"
#include "commons/utils/SetuCommHelper.h"
#include "commons/utils/ThreadingUtils.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::enums::ErrorCode;
using setu::commons::messages::ExecuteProgramRequest;
using setu::commons::messages::ExecuteProgramResponse;
using setu::commons::utils::SetuCommHelper;
using setu::commons::utils::ZmqHelper;
//==============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    ASSERT_VALID_RUNTIME(err == cudaSuccess, "CUDA error: {} at {}:{}",        \
                         cudaGetErrorString(err), __FILE__, __LINE__);         \
  } while (0)

#define NCCL_CHECK(call)                                                       \
  do {                                                                         \
    ncclResult_t res = (call);                                                 \
    ASSERT_VALID_RUNTIME(res == ncclSuccess, "NCCL error: {} at {}:{}",        \
                         ncclGetErrorString(res), __FILE__, __LINE__);         \
  } while (0)


//==============================================================================
// NCCLWorker
//==============================================================================

NCCLWorker::NCCLWorker(Device device,
                       std::size_t reply_port)
    : Worker(device, reply_port),
      stream_(nullptr) {}

NCCLWorker::~NCCLWorker() {
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
  for (auto& [key, entry] : comm_cache_) {
    ncclCommDestroy(entry.nccl_comm);
  }
}

void NCCLWorker::Setup() {
  CUDA_CHECK(cudaSetDevice(device_.local_device_rank));
  CUDA_CHECK(cudaStreamCreate(&stream_));
  LOG_DEBUG("NCCLWorker setup complete for device {}", device_);
}

void NCCLWorker::Execute(const Program& program) {
  LOG_DEBUG("Executing program with {} instructions", program.instrs.size());

  bool group_started = false;

  for (const auto& instruction : program.instrs) {
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

        if constexpr (std::is_same_v<T, InitCommInstruction>) {
          ExecuteInitComm(inst);
        } else if constexpr (std::is_same_v<T, UseCommInstruction>) {
          ExecuteUseComm(inst);
        } else if constexpr (std::is_same_v<T, CopyInstruction>) {
          if (!group_started) {
            NCCL_CHECK(ncclGroupStart());
            group_started = true;
          }
          ExecuteCopy(inst);
        } else if constexpr (std::is_same_v<T, SendInstruction>) {
          if (!group_started) {
            NCCL_CHECK(ncclGroupStart());
            group_started = true;
          }
          ExecuteSend(inst);
        } else if constexpr (std::is_same_v<T, ReceiveInstruction>) {
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

void NCCLWorker::ExecuteInitComm(const InitCommInstruction& inst) {
  std::string key = CommIdToString(inst.comm_id);

  const std::int32_t num_ranks =
      static_cast<std::int32_t>(inst.device_to_rank.size());
  const std::int32_t rank = inst.device_to_rank.at(device_.device_rank);

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, inst.comm_id, rank));

  comm_cache_[key] = CommCacheEntry{
      .nccl_comm = comm,
      .device_to_rank = inst.device_to_rank,
  };

  active_comm_key_ = key;
  LOG_DEBUG("InitComm complete: {} ranks, this rank={}", num_ranks, rank);
}

void NCCLWorker::ExecuteUseComm(const UseCommInstruction& inst) {
  active_comm_key_ = CommIdToString(inst.comm_id);
  LOG_DEBUG("UseComm: switched to communicator");
}

void NCCLWorker::ExecuteCopy(const CopyInstruction& inst) {
  const auto& [src_tensor_name, src_shard_id] = inst.src_tensor;
  const auto& [dst_tensor_name, dst_shard_id] = inst.dst_tensor;

  const std::size_t bytes = inst.num_elements * GetDTypeSizeBytes(inst.dtype);

  CUDA_CHECK(cudaMemcpyAsync(
      static_cast<char*>(inst.dst_ptr) + inst.dst_memory_offset_bytes,
      static_cast<char*>(inst.src_ptr) + inst.src_memory_offset_bytes, bytes,
      cudaMemcpyDeviceToDevice, stream_));

  LOG_DEBUG("Copy: {} bytes from {}:{} to {}:{}", bytes, src_tensor_name,
            src_shard_id, dst_tensor_name, dst_shard_id);
}

void NCCLWorker::ExecuteSend(const SendInstruction& inst) {
  auto& entry = comm_cache_.at(active_comm_key_);
  const std::int32_t peer_rank = entry.device_to_rank.at(inst.dst_device_id);

  const auto& [tensor_name, shard_id] = inst.src_tensor;

  NCCL_CHECK(ncclSend(
      static_cast<char*>(inst.src_ptr) + inst.memory_offset_bytes,
      inst.num_elements, ToNcclDataType(inst.dtype), peer_rank, entry.nccl_comm,
      stream_));

  LOG_DEBUG("Send: {} elements from {}:{} to device: {}", inst.num_elements, tensor_name, shard_id, peer_rank);
}

void NCCLWorker::ExecuteReceive(const ReceiveInstruction& inst) {
  auto& entry = comm_cache_.at(active_comm_key_);
  const std::int32_t peer_rank = entry.device_to_rank.at(inst.src_device_id);

  const auto& [tensor_name, shard_id] = inst.dst_tensor;

  NCCL_CHECK(ncclRecv(static_cast<char*>(inst.dst_ptr) + inst.memory_offset_bytes,
                      inst.num_elements, ToNcclDataType(inst.dtype), peer_rank,
                      entry.nccl_comm, stream_));

  LOG_DEBUG("Receive: {} elements from device: {}:{} from device: {}", inst.num_elements, tensor_name, shard_id, peer_rank);
}

//==============================================================================
// Helper Functions
//==============================================================================

std::string NCCLWorker::CommIdToString(const ncclUniqueId& id) {
  return std::string(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}

ncclDataType_t NCCLWorker::ToNcclDataType(DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
      return ncclFloat16;
    case DType::kBFloat16:
      return ncclBfloat16;
    case DType::kFloat32:
      return ncclFloat32;
    default:
      RAISE_RUNTIME_ERROR("Unsupported dtype: {}", static_cast<int>(dtype));
  }
}

std::size_t NCCLWorker::GetDTypeSizeBytes(DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
      return 2;
    case DType::kBFloat16:
      return 2;
    case DType::kFloat32:
      return 4;
    default:
      RAISE_RUNTIME_ERROR("Unsupported dtype: {}", static_cast<int>(dtype));
  }
}

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
