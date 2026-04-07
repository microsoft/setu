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
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/CUDAUtils.h"
#include "commons/utils/EnvUtils.h"
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
  using setu::commons::utils::GetEnv;
  CUDA_CHECK(cudaSetDevice(device_.LocalDeviceIndex()));

  const auto num_streams =
      GetEnv<std::size_t>("SETU_WORKER_NUM_STREAMS", kDefaultNumStreams);
  ASSERT_VALID_ARGUMENTS(num_streams >= 1, "SETU_WORKER_NUM_STREAMS must be >= 1");

  max_in_flight_ =
      GetEnv<std::size_t>("SETU_WORKER_MAX_INFLIGHT_PLANS", kDefaultMaxInFlight);
  ASSERT_VALID_ARGUMENTS(max_in_flight_ >= 1,
                         "SETU_WORKER_MAX_INFLIGHT_PLANS must be >= 1");

  LOG_INFO("NCCLWorker[{}]: num_streams={}, max_in_flight={}",
           device_, num_streams, max_in_flight_);

  streams_.resize(num_streams);
  stream_loads_.resize(num_streams, 0);
  for (auto& s : streams_) {
    CUDA_CHECK(cudaStreamCreate(&s));
  }
  active_stream_ = streams_[0];

  // Allocate completion event ring for async dispatch
  completion_event_ring_.resize(max_in_flight_);
  for (auto& event_set : completion_event_ring_) {
    event_set.resize(num_streams, nullptr);
    for (auto& e : event_set) {
      CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
  }

  // Name CUDA streams and this thread for NVTX profiling.
  for (std::size_t i = 0; i < streams_.size(); ++i) {
    auto name =
        std::format("setu::worker[gpu{}]::stream{}",
                    device_.LocalDeviceIndex(), i);
    nvtxNameCudaStreamA(streams_[i], name.c_str());
  }
  nvtxNameOsThread(
      pthread_self(),
      std::format("setu::worker[gpu{}]", device_.LocalDeviceIndex()).c_str());

  if (!register_file_.Empty()) {
    register_file_.Allocate();
    LOG_DEBUG("Allocated {} registers on device {}",
              register_file_.NumRegisters(), device_);
  }

  LOG_DEBUG("NCCLWorker setup complete for device {}", device_);
}

void NCCLWorker::Execute(const Program& program) {
  if (has_last_execute_end_) {
    auto gap_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - last_execute_end_).count();
    LOG_DEBUG("PIPELINE[{}]: inter_execute_gap={}us copy_op={}",
              device_, gap_us, current_copy_op_id_);
  }

  auto nvtx_label = std::format("CopyOp:{} dev:{}",
      boost::uuids::to_string(current_copy_op_id_), device_);
  nvtxEventAttributes_t nvtx_attr = {};
  nvtx_attr.version = NVTX_VERSION;
  nvtx_attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  nvtx_attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_attr.message.ascii = nvtx_label.c_str();
  nvtx_attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  nvtx_attr.payload.ullValue = program.size();
  nvtxRangePushEx(&nvtx_attr);

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

  // Reset stream loads for this execution.
  std::fill(stream_loads_.begin(), stream_loads_.end(), 0);

  // Select the least-loaded stream and charge it with op_bytes.
  auto select_stream = [&](std::size_t op_bytes) {
    auto idx = LeastLoadedStream();
    stream_loads_[idx] += op_bytes;
    active_stream_ = streams_[idx];
  };

  // Reset event pool mappings for this program.
  event_pool_.Reset();

  auto apply_pending_waits = [&]() {
    for (auto event : pending_waits_) {
      CUDA_CHECK(cudaStreamWaitEvent(active_stream_, event));
    }
    pending_waits_.clear();
  };

  for (const auto& instruction : program) {
    std::visit(
        [&](const auto& inst) {
          using T = std::decay_t<decltype(inst)>;

          if constexpr (std::is_same_v<T, InitComm>) {
            ExecuteInitComm(inst);
          } else if constexpr (std::is_same_v<T, Send>) {
            select_stream(inst.count * GetDTypeSizeBytes(inst.dtype));
            apply_pending_waits();
            ExecuteSend(inst);
          } else if constexpr (std::is_same_v<T, Receive>) {
            select_stream(inst.count * GetDTypeSizeBytes(inst.dtype));
            apply_pending_waits();
            ExecuteReceive(inst);
          } else if constexpr (std::is_same_v<T, Copy>) {
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
            ExecuteAllGather(inst);
          } else if constexpr (std::is_same_v<T, Fence>) {
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
  ring_head_ = (ring_head_ + 1) % max_in_flight_;

  auto t_end = std::chrono::high_resolution_clock::now();
  double dispatch_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  LOG_DEBUG("Execute[{}]: dispatched copy_op={}, {}ms host time, {} streams",
            device_, current_copy_op_id_, dispatch_ms, num_streams_used);

  last_execute_end_ = std::chrono::steady_clock::now();
  has_last_execute_end_ = true;

  nvtxRangePop();
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
    nvtxMarkA(std::format("complete::{}",
                          boost::uuids::to_string(oldest.copy_op_id))
                  .c_str());
    LOG_DEBUG("DrainCompletions[{}]: copy_op={} complete",
              device_, oldest.copy_op_id);
    completion_queue_->push(
        WorkerCompletion{oldest.copy_op_id, device_.LocalDeviceIndex()});
    pending_programs_.pop_front();
  }
}

void NCCLWorker::WaitForCapacity() {
  while (pending_programs_.size() >= max_in_flight_) {
    DrainCompletions();
    if (pending_programs_.size() >= max_in_flight_) {
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

  std::vector<const void*> srcs(count);
  std::vector<void*> dsts(count);
  std::vector<std::size_t> sizes(count);

  for (std::size_t i = 0; i < count; ++i) {
    const auto& e = inst.entries[i];
    srcs[i] = static_cast<char*>(e.src_ptr) + e.src_offset_bytes;
    dsts[i] = static_cast<char*>(e.dst_ptr) + e.dst_offset_bytes;
    sizes[i] = e.count * GetDTypeSizeBytes(e.dtype);
  }

  cudaMemcpyAttributes attrs = {};
  attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  std::size_t fail_idx = 0;
#if CUDART_VERSION >= 13000
  // CUDA 13+: 8-arg signature with (attrs*, attrsIdxs*, numAttrs, stream)
  CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), count,
                                  &attrs, &fail_idx,
                                  static_cast<std::size_t>(1), active_stream_));
#else
  // CUDA 12.x: 7-arg template with (attrs_by_value, failIdxs*, stream)
  CUDA_CHECK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), count,
                                  attrs, &fail_idx, active_stream_));
#endif

  LOG_DEBUG("Copy: {} entries batched via cudaMemcpyBatchAsync", count);
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
  auto event = event_pool_.Acquire(inst.id);
  CUDA_CHECK(cudaEventRecord(event, active_stream_));
  LOG_DEBUG("SyncPoint({}): recorded event on stream", inst.id);
}

void NCCLWorker::ExecuteWait(const Wait& inst) {
  auto event = event_pool_.Get(inst.id);
  if (event != nullptr) {
    pending_waits_.push_back(event);
    LOG_DEBUG("Wait({}): buffered dependency", inst.id);
  } else {
    LOG_DEBUG("Wait({}): skipped, event already completed", inst.id);
  }
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
