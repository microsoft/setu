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
#pragma once
//==============================================================================
#include <cuda_runtime.h>
#include <nccl.h>
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/Device.h"
#include "commons/enums/Enums.h"
#include "node_manager/worker/RegisterFile.h"
#include "node_manager/worker/Worker.h"
#include "planner/Constants.h"
#include "planner/ir/llc/CommId.h"
#include "planner/ir/llc/Instruction.h"
#include "planner/ir/llc/instructions/SyncPoint.h"
#include "planner/ir/llc/instructions/Wait.h"
#include "telemetry/NCCLWorkerMetrics.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::NodeId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::Device;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
using setu::planner::ir::llc::AllGather;
using setu::planner::ir::llc::CommId;
using setu::planner::ir::llc::CommIdHash;
using setu::planner::ir::llc::Copy;
using setu::planner::ir::llc::Fence;
using setu::planner::ir::llc::InitComm;
using setu::planner::ir::llc::Instruction;
using setu::planner::ir::llc::Program;
using setu::planner::ir::llc::Receive;
using setu::planner::ir::llc::Send;
using setu::planner::ir::llc::SyncPoint;
using setu::planner::ir::llc::Wait;
//==============================================================================

class NCCLWorker : public Worker {
 public:
  NCCLWorker(NodeId node_id, Device device,
             RegisterSet register_set =
                 RegisterSet::Uniform(1, setu::planner::kRegisterSize));
  ~NCCLWorker();

  void Execute(const Program& program) override;
  void Setup() override;

  [[nodiscard]] DevicePtr ResolveRegister(
      const RegisterRef& ref) const override;

 private:
  void ExecuteInitComm(const InitComm& inst);
  void ExecuteCopy(const Copy& inst);
  void ExecuteSend(const Send& inst);
  void ExecuteReceive(const Receive& inst);
  void ExecuteAllGather(const AllGather& inst);
  void ExecuteSyncPoint(const SyncPoint& inst);
  void ExecuteWait(const Wait& inst);

  [[nodiscard]] static ncclDataType_t ToNcclDataType(torch::Dtype dtype);
  [[nodiscard]] static std::size_t GetDTypeSizeBytes(torch::Dtype dtype);

  struct CommCacheEntry {
    ncclComm_t nccl_comm;
  };

  std::unordered_map<CommId, CommCacheEntry, CommIdHash> comm_cache_;

  /// Pool of CUDA streams for overlapping independent operations.
  /// Round-robin assigned per op; independent ops naturally land on
  /// different streams without any explicit reset.
  std::vector<cudaStream_t> streams_;
  cudaStream_t active_stream_ = nullptr;

  /// Lazily grown pool of CUDA events, indexed by SyncPoint id.
  /// Events are created on first use and reused across Execute() calls.
  std::vector<cudaEvent_t> event_pool_;

  /// Two pre-allocated CUDA events for measuring total execution time.
  cudaEvent_t timing_start_ = nullptr;
  cudaEvent_t timing_end_ = nullptr;

  /// Buffered Wait ids accumulated between data ops.
  /// Flushed as cudaStreamWaitEvent calls at the start of the next data op.
  std::vector<std::uint32_t> pending_waits_;

  RegisterFile register_file_;
};

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
