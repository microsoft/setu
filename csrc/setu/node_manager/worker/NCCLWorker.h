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
#include "commons/utils/ZmqHelper.h"
#include "node_manager/worker/RegisterFile.h"
#include "node_manager/worker/Worker.h"
#include "planner/Constants.h"
#include "planner/ir/llc/Instruction.h"
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
using setu::planner::ir::llc::Barrier;
using setu::planner::ir::llc::Copy;
using setu::planner::ir::llc::InitComm;
using setu::planner::ir::llc::Instruction;
using setu::planner::ir::llc::Program;
using setu::planner::ir::llc::Receive;
using setu::planner::ir::llc::Send;
using setu::planner::ir::llc::UseComm;
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
  void ExecuteInstruction(const Instruction& instruction, bool& group_started);

  void ExecuteInitComm(const InitComm& inst);
  void ExecuteUseComm(const UseComm& inst);
  void ExecuteCopy(const Copy& inst);
  void ExecuteSend(const Send& inst);
  void ExecuteReceive(const Receive& inst);

  [[nodiscard]] static std::string CommIdToString(const ncclUniqueId& id);
  [[nodiscard]] static ncclDataType_t ToNcclDataType(torch::Dtype dtype);
  [[nodiscard]] static std::size_t GetDTypeSizeBytes(torch::Dtype dtype);

  struct CommCacheEntry {
    ncclComm_t nccl_comm;
  };

  std::unordered_map<std::string, CommCacheEntry> comm_cache_;
  std::string active_comm_key_;
  cudaStream_t stream_;

  RegisterFile register_file_;
};

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
