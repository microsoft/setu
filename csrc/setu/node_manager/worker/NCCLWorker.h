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
#include "ir/Instruction.h"
#include "node_manager/worker/Worker.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::Device;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
using setu::ir::CopyInstruction;
using setu::ir::InitCommInstruction;
using setu::ir::Instruction;
using setu::ir::Program;
using setu::ir::ReceiveInstruction;
using setu::ir::SendInstruction;
using setu::ir::UseCommInstruction;
//==============================================================================

class NCCLWorker : public Worker {
 public:
  NCCLWorker(Device device, std::size_t reply_port);
  ~NCCLWorker();

  void Execute(const Program& program) override;
  void Setup() override;

 private:
  void ExecuteInstruction(const Instruction& instruction, bool& group_started);

  void ExecuteInitComm(const InitCommInstruction& inst);
  void ExecuteUseComm(const UseCommInstruction& inst);
  void ExecuteCopy(const CopyInstruction& inst);
  void ExecuteSend(const SendInstruction& inst);
  void ExecuteReceive(const ReceiveInstruction& inst);

  [[nodiscard]] static std::string CommIdToString(const ncclUniqueId& id);
  [[nodiscard]] static ncclDataType_t ToNcclDataType(torch::Dtype dtype);
  [[nodiscard]] static std::size_t GetDTypeSizeBytes(torch::Dtype dtype);

  struct CommCacheEntry {
    ncclComm_t nccl_comm;
    std::unordered_map<DeviceRank, std::int32_t> device_to_rank;
  };

  std::unordered_map<std::string, CommCacheEntry> comm_cache_;
  std::string active_comm_key_;
  cudaStream_t stream_;
};

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
