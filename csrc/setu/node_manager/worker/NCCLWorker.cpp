//=============================================================================
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
#include "commons/utils/CUDAUtils.h"
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
// NCCLWorker
//==============================================================================

NCCLWorker::NCCLWorker(NodeId node_id, Device device, RegisterSet register_set)
    : Worker(node_id, device), register_file_(std::move(register_set)) {}

NCCLWorker::~NCCLWorker() {
  for (auto s : streams_) {
    cudaStreamDestroy(s);
  }
  for (auto& [key, entry] : comm_cache_) {
    ncclCommDestroy(entry.nccl_comm);
  }
}

void NCCLWorker::Setup() {
  CUDA_CHECK(cudaSetDevice(device_.LocalDeviceIndex()));
  static constexpr std::size_t kNumStreams = 1024;
  streams_.resize(kNumStreams);
  for (auto& s : streams_) {
    CUDA_CHECK(cudaStreamCreate(&s));
  }
  active_stream_ = streams_[0];

  if (!register_file_.Empty()) {
    register_file_.Allocate();
    LOG_DEBUG("Allocated {} registers on device {}",
              register_file_.NumRegisters(), device_);
  }

  LOG_DEBUG("NCCLWorker setup complete for device {}", device_);
}

void NCCLWorker::Execute(const Program& program) {
  auto t_start = std::chrono::high_resolution_clock::now();

  std::uint32_t group_index = 0;
  std::vector<GroupTimingState> event_states;
  std::vector<setu::telemetry::NCCLGroupTiming> timings;

  bool in_nccl_group = false;

  auto open_nccl_group = [&]() {
    if (!in_nccl_group) {
      NCCL_CHECK(ncclGroupStart());
      in_nccl_group = true;
    }
  };

  auto close_nccl_group = [&]() {
    if (in_nccl_group) {
      NCCL_CHECK(ncclGroupEnd());
      in_nccl_group = false;
    }
  };

  for (const auto& instruction : program) {
    std::visit(
        [&](const auto& inst) {
          using T = std::decay_t<decltype(inst)>;

          if constexpr (std::is_same_v<T, InitComm>) {
            close_nccl_group();
            ExecuteInitComm(inst);
          } else if constexpr (std::is_same_v<T, Copy> ||
                               std::is_same_v<T, Send> ||
                               std::is_same_v<T, Receive> ||
                               std::is_same_v<T, AllGather>) {
            active_stream_ = streams_[group_index % streams_.size()];

            cudaEvent_t start_event;
            CUDA_CHECK(cudaEventCreate(&start_event));
            CUDA_CHECK(cudaEventRecord(start_event, active_stream_));

            open_nccl_group();

            if constexpr (std::is_same_v<T, Copy>) {
              ExecuteCopy(inst);
            } else if constexpr (std::is_same_v<T, Send>) {
              ExecuteSend(inst);
            } else if constexpr (std::is_same_v<T, Receive>) {
              ExecuteReceive(inst);
            } else {
              ExecuteAllGather(inst);
            }

            cudaEvent_t end_event;
            CUDA_CHECK(cudaEventCreate(&end_event));
            CUDA_CHECK(cudaEventRecord(end_event, active_stream_));
            event_states.push_back({start_event, end_event, 1});
            timings.push_back({group_index, 0.0, 1});
            group_index++;
          } else if constexpr (std::is_same_v<T, Fence>) {
            close_nccl_group();
            const std::size_t num_used = std::min(
                static_cast<std::size_t>(group_index), streams_.size());
            if (num_used > 0) {
              std::vector<cudaEvent_t> fence_events(num_used);
              for (std::size_t i = 0; i < num_used; ++i) {
                CUDA_CHECK(cudaEventCreate(&fence_events[i]));
                CUDA_CHECK(cudaEventRecord(fence_events[i], streams_[i]));
              }
              for (std::size_t i = 0; i < num_used; ++i) {
                for (std::size_t j = 0; j < num_used; ++j) {
                  if (i != j) {
                    CUDA_CHECK(
                        cudaStreamWaitEvent(streams_[i], fence_events[j]));
                  }
                }
              }
              for (auto& e : fence_events) {
                CUDA_CHECK(cudaEventDestroy(e));
              }
            }
            group_index = 0;
          }
        },
        instruction.instr);
  }

  // Close any trailing group
  close_nccl_group();

  // Sync only streams that had work queued.
  const std::size_t num_streams_used =
      std::min(static_cast<std::size_t>(group_index) + 1, streams_.size());
  for (std::size_t i = 0; i < num_streams_used; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
  }

  // Compute elapsed times from CUDA event pairs
  for (std::size_t i = 0; i < event_states.size(); ++i) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, event_states[i].start_event,
                                    event_states[i].end_event));
    timings[i].elapsed_ms = static_cast<double>(ms);
    CUDA_CHECK(cudaEventDestroy(event_states[i].start_event));
    CUDA_CHECK(cudaEventDestroy(event_states[i].end_event));
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  // Submit metrics if sink is available
  if (metrics_sink_ && metrics_sink_->IsEnabled()) {
    setu::telemetry::NCCLWorkerMetrics wm;
    wm.copy_op_id = current_copy_op_id_;
    wm.node_id = node_id_;
    wm.device_rank = device_.LocalDeviceIndex();
    wm.group_timings = std::move(timings);
    wm.total_execute_ms = total_ms;
    metrics_sink_->Submit(setu::telemetry::MetricsMessage{wm});
  }
}

//==============================================================================
// Instruction Handlers
//==============================================================================

void NCCLWorker::ExecuteInitComm(const InitComm& inst) {
  const std::int32_t num_ranks =
      static_cast<std::int32_t>(inst.participant_to_rank.size());
  auto part = Participant(node_id_, device_);
  const std::int32_t rank = inst.participant_to_rank.at(part);

  auto nccl_id = inst.comm_id.As<ncclUniqueId>();

  auto t0 = std::chrono::steady_clock::now();
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, nccl_id, rank));
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0)
                .count();

  comm_cache_[inst.comm_id] = CommCacheEntry{.nccl_comm = comm};

  LOG_INFO("InitComm[{}]: ncclCommInitRank took {}ms, {} ranks, this rank={}",
           device_, dt, num_ranks, rank);
}

void NCCLWorker::ExecuteCopy(const Copy& inst) {
  ASSERT_VALID_RUNTIME(!inst.entries.empty(),
                       "Copy instruction must have at least one entry");

  const std::size_t count = inst.entries.size();

  std::vector<void*> srcs(count);
  std::vector<void*> dsts(count);
  std::vector<std::size_t> sizes(count);

  for (std::size_t i = 0; i < count; ++i) {
    const auto& e = inst.entries[i];
    srcs[i] = static_cast<char*>(e.src_ptr) + e.src_offset_bytes;
    dsts[i] = static_cast<char*>(e.dst_ptr) + e.dst_offset_bytes;
    sizes[i] = e.count * GetDTypeSizeBytes(e.dtype);
  }

  for (std::size_t i = 0; i < count; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i],
                               cudaMemcpyDeviceToDevice, active_stream_));
  }

  LOG_DEBUG("Copy: {} entries via cudaMemcpyAsync", count);
}

void NCCLWorker::ExecuteSend(const Send& inst) {
  auto& entry = comm_cache_.at(inst.comm_id);

  NCCL_CHECK(ncclSend(static_cast<char*>(inst.src_ptr) + inst.offset_bytes,
                      inst.count, ToNcclDataType(inst.dtype), inst.peer_rank,
                      entry.nccl_comm, active_stream_));

  LOG_DEBUG("Send: {} elements from {} to device rank: {}", inst.count,
            inst.src_ref.ToString(), inst.peer_rank);
}

void NCCLWorker::ExecuteReceive(const Receive& inst) {
  auto& entry = comm_cache_.at(inst.comm_id);

  NCCL_CHECK(ncclRecv(static_cast<char*>(inst.dst_ptr) + inst.offset_bytes,
                      inst.count, ToNcclDataType(inst.dtype), inst.peer_rank,
                      entry.nccl_comm, active_stream_));

  LOG_DEBUG("Receive: {} elements to {} from device rank: {}", inst.count,
            inst.dst_ref.ToString(), inst.peer_rank);
}

void NCCLWorker::ExecuteAllGather(const AllGather& inst) {
  auto& entry = comm_cache_.at(inst.comm_id);

  auto* send_buf = static_cast<char*>(inst.send_ptr) + inst.send_offset_bytes;
  auto* recv_buf = static_cast<char*>(inst.recv_ptr) + inst.recv_offset_bytes;

  NCCL_CHECK(ncclAllGather(send_buf, recv_buf, inst.send_count,
                           ToNcclDataType(inst.dtype), entry.nccl_comm,
                           active_stream_));

  LOG_DEBUG("AllGather: {} elements/rank, {} ranks, send={}, recv={}",
            inst.send_count, inst.num_ranks, inst.send_ref.ToString(),
            inst.recv_ref.ToString());
}

DevicePtr NCCLWorker::ResolveRegister(const RegisterRef& ref) const {
  return register_file_.GetPtr(ref.register_index);
}

//==============================================================================
// Helper Functions
//==============================================================================

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
