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
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
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
  for (auto e : event_pool_) {
    cudaEventDestroy(e);
  }
  for (auto& event_set : completion_event_ring_) {
    for (auto& e : event_set) {
      if (e) cudaEventDestroy(e);
    }
  }
  for (auto& [key, entry] : comm_cache_) {
    ncclCommDestroy(entry.nccl_comm);
  }
}

void NCCLWorker::Setup() {
  CUDA_CHECK(cudaSetDevice(device_.LocalDeviceIndex()));
  static const std::size_t kNumStreams = []() -> std::size_t {
    const char* env = std::getenv("SETU_NCCL_NUM_STREAMS");
    if (env != nullptr) {
      auto val = std::stoul(env);
      ASSERT_VALID_ARGUMENTS(val >= 1, "SETU_NCCL_NUM_STREAMS must be >= 1");
      return val;
    }
    return 2;
  }();
  streams_.resize(kNumStreams);
  stream_loads_.resize(kNumStreams, 0);
  for (auto& s : streams_) {
    CUDA_CHECK(cudaStreamCreate(&s));
  }
  active_stream_ = streams_[0];

  // Allocate completion event ring for async dispatch
  completion_event_ring_.resize(kMaxInFlight);
  for (auto& event_set : completion_event_ring_) {
    event_set.resize(kNumStreams, nullptr);
    for (auto& e : event_set) {
      CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
  }

  if (!register_file_.Empty()) {
    register_file_.Allocate();
    LOG_DEBUG("Allocated {} registers on device {}",
              register_file_.NumRegisters(), device_);
  }

  LOG_DEBUG("NCCLWorker setup complete for device {}", device_);
}

void NCCLWorker::Execute(const Program& program) {
  auto t_start = std::chrono::high_resolution_clock::now();

  //============================================================================
  // Inter-program fence: make all streams wait on the previous program's
  // boundary events. This ensures plans finish in GPU order and prevents
  // concurrent NCCL spinning kernels from different communicators.
  //============================================================================
  if (!pending_programs_.empty()) {
    const auto& prev = pending_programs_.back();
    for (std::size_t i = 0; i < prev.num_streams_used; ++i) {
      for (auto& s : streams_) {
        CUDA_CHECK(cudaStreamWaitEvent(
            s, completion_event_ring_[prev.ring_slot][i]));
      }
    }
  }

  //============================================================================
  // Instruction dispatch
  //============================================================================
  bool in_nccl_group = false;

  auto open_nccl_group = [&]() {
    if (!in_nccl_group) {
      NCCL_CHECK(ncclGroupStart());
      in_nccl_group = true;
    }
  };

  std::int64_t total_group_end_us = 0;
  std::uint32_t group_end_count = 0;

  auto close_nccl_group = [&]() {
    if (in_nccl_group) {
      auto t0 = std::chrono::steady_clock::now();
      NCCL_CHECK(ncclGroupEnd());
      auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
      total_group_end_us += dt;
      group_end_count++;
      in_nccl_group = false;
    }
  };

  // Reset stream loads for this execution.
  std::fill(stream_loads_.begin(), stream_loads_.end(), 0);

  // Select the least-loaded stream and charge it with op_bytes.
  auto select_stream = [&](std::size_t op_bytes) {
    auto idx = LeastLoadedStream();
    stream_loads_[idx] += op_bytes;
    active_stream_ = streams_[idx];
  };

  auto apply_pending_waits = [&]() {
    for (auto wait_id : pending_waits_) {
      ASSERT_VALID_RUNTIME(
          wait_id < event_pool_.size(),
          "Wait references event id {} but event pool size is {}",
          wait_id, event_pool_.size());
      CUDA_CHECK(
          cudaStreamWaitEvent(active_stream_, event_pool_[wait_id]));
    }
    pending_waits_.clear();
  };

  for (const auto& instruction : program) {
    std::visit(
        [&](const auto& inst) {
          using T = std::decay_t<decltype(inst)>;

          if constexpr (std::is_same_v<T, InitComm>) {
            close_nccl_group();
            ExecuteInitComm(inst);
          } else if constexpr (std::is_same_v<T, Send>) {
            close_nccl_group();
            select_stream(inst.count * GetDTypeSizeBytes(inst.dtype));
            apply_pending_waits();
            ExecuteSend(inst);
          } else if constexpr (std::is_same_v<T, Receive>) {
            close_nccl_group();
            select_stream(inst.count * GetDTypeSizeBytes(inst.dtype));
            apply_pending_waits();
            ExecuteReceive(inst);
          } else if constexpr (std::is_same_v<T, Copy>) {
            close_nccl_group();
            std::size_t total_bytes = 0;
            for (const auto& e : inst.entries) {
              total_bytes += e.count * GetDTypeSizeBytes(e.dtype);
            }
            select_stream(total_bytes);
            apply_pending_waits();
            ExecuteCopy(inst);
          } else if constexpr (std::is_same_v<T, AllGather>) {
            select_stream(inst.send_count * GetDTypeSizeBytes(inst.dtype));
            apply_pending_waits();
            open_nccl_group();
            ExecuteAllGather(inst);
          } else if constexpr (std::is_same_v<T, Fence>) {
            close_nccl_group();
            // Cross-synchronize all streams.
            std::vector<cudaEvent_t> fence_events(streams_.size());
            for (std::size_t i = 0; i < streams_.size(); ++i) {
              CUDA_CHECK(cudaEventCreate(&fence_events[i]));
              CUDA_CHECK(cudaEventRecord(fence_events[i], streams_[i]));
            }
            for (std::size_t i = 0; i < streams_.size(); ++i) {
              for (std::size_t j = 0; j < streams_.size(); ++j) {
                if (i != j) {
                  CUDA_CHECK(
                      cudaStreamWaitEvent(streams_[i], fence_events[j]));
                }
              }
            }
            for (auto& e : fence_events) {
              CUDA_CHECK(cudaEventDestroy(e));
            }
            // Reset loads after a fence.
            std::fill(stream_loads_.begin(), stream_loads_.end(), 0);
          } else if constexpr (std::is_same_v<T, SyncPoint>) {
            ExecuteSyncPoint(inst);
          } else if constexpr (std::is_same_v<T, Wait>) {
            ExecuteWait(inst);
          }
        },
        instruction.instr);
  }

  close_nccl_group();

  LOG_INFO("Execute[{}]: {} ncclGroupEnd calls, total {}us", device_,
           group_end_count, total_group_end_us);

  //============================================================================
  // Record boundary events on all used streams (replaces cudaStreamSynchronize)
  //============================================================================
  std::size_t num_streams_used = 0;
  const std::size_t slot = ring_head_;
  for (std::size_t i = 0; i < streams_.size(); ++i) {
    if (stream_loads_[i] > 0) {
      CUDA_CHECK(cudaEventRecord(completion_event_ring_[slot][num_streams_used],
                                 streams_[i]));
      ++num_streams_used;
    }
  }

  pending_programs_.push_back(PendingProgram{
      current_copy_op_id_,
      static_cast<std::uint32_t>(num_streams_used),
      slot});
  ring_head_ = (ring_head_ + 1) % kMaxInFlight;

  auto t_end = std::chrono::high_resolution_clock::now();
  double dispatch_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  LOG_DEBUG("Execute[{}]: dispatched copy_op={}, {}ms host time, {} streams",
            device_, current_copy_op_id_, dispatch_ms, num_streams_used);
}

//==============================================================================
// Async completion hooks
//==============================================================================

bool NCCLWorker::HasPendingCompletions() const {
  return !pending_programs_.empty();
}

void NCCLWorker::DrainCompletions() {
  while (!pending_programs_.empty()) {
    const auto& oldest = pending_programs_.front();

    bool all_done = true;
    for (std::uint32_t i = 0; i < oldest.num_streams_used; ++i) {
      cudaError_t status =
          cudaEventQuery(completion_event_ring_[oldest.ring_slot][i]);
      if (status == cudaErrorNotReady) {
        all_done = false;
        break;
      }
      CUDA_CHECK(status);
    }

    if (!all_done) break;

    // All streams for this program are done on the GPU
    LOG_DEBUG("DrainCompletions[{}]: copy_op={} complete",
              device_, oldest.copy_op_id);
    completion_queue_->push(
        WorkerCompletion{oldest.copy_op_id, device_.LocalDeviceIndex()});
    pending_programs_.pop_front();
  }
}

void NCCLWorker::WaitForCapacity() {
  while (pending_programs_.size() >= kMaxInFlight) {
    DrainCompletions();
    if (pending_programs_.size() >= kMaxInFlight) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
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

void NCCLWorker::ExecuteSyncPoint(const SyncPoint& inst) {
  // Grow the event pool lazily to accommodate this id.
  if (event_pool_.size() <= inst.id) {
    auto t0 = std::chrono::steady_clock::now();
    while (event_pool_.size() <= inst.id) {
      cudaEvent_t e;
      CUDA_CHECK(cudaEventCreate(&e));
      event_pool_.push_back(e);
    }
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
    LOG_INFO("SyncPoint({}): grew event pool to {} events in {}us", inst.id,
             event_pool_.size(), dt);
  }
  // Record on the stream that just ran the preceding write op.
  // active_stream_ is still set to that op's stream at this point.
  CUDA_CHECK(cudaEventRecord(event_pool_[inst.id], active_stream_));
  LOG_DEBUG("SyncPoint({}): recorded event on stream", inst.id);
}

void NCCLWorker::ExecuteWait(const Wait& inst) {
  // Buffer the id; the dependency will be applied to the next data op's
  // stream when that op sets active_stream_.
  pending_waits_.push_back(inst.id);
  LOG_DEBUG("Wait({}): buffered dependency", inst.id);
}

DevicePtr NCCLWorker::ResolveRegister(const RegisterRef& ref) const {
  return register_file_.GetPtr(ref.register_index);
}

std::size_t NCCLWorker::LeastLoadedStream() const {
  std::size_t best = 0;
  for (std::size_t i = 1; i < stream_loads_.size(); ++i) {
    if (stream_loads_[i] < stream_loads_[best]) {
      best = i;
    }
  }
  return best;
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
