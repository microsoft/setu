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
#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/Device.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/enums/Enums.h"
#include "commons/utils/ZmqHelper.h"
#include "ir/Instruction.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::enums::ErrorCode;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
using setu::ir::Instruction;
using setu::ir::Program;
//==============================================================================
class Worker {
 public:
  Worker(Device device, std::size_t port);
  ~Worker();

  void Start();
  void Stop();

  [[nodiscard]] bool IsRunning() const { return worker_running_.load(); }

  [[nodiscard]] const Device& GetDevice() const { return device_; }

  void Execute(const Program& instrs);

 private:
  void InitZmqSockets();
  void CloseZmqSockets();

  void StartExecutorLoop();
  void StopExecutorLoop();

  void ExecutorLoop();
  void ExecuteInstruction(const Instruction& instruction);

  Device device_;

  std::size_t port_;
  ZmqContextPtr zmq_context_;
  ZmqSocketPtr socket_;

  std::atomic<bool> worker_running_{false};

  std::thread executor_thread_;
};
//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
