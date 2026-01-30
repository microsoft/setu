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
#include "coordinator/datatypes/Instruction.h"
#include "coordinator/datatypes/Program.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::ClientRank;
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::CopyOperationId;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::enums::ErrorCode;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
using setu::coordinator::datatypes::Instruction;
using setu::coordinator::datatypes::CopyInstruction;
using setu::coordinator::datatypes::ReceiveInstruction;
using setu::coordinator::datatypes::SendInstruction;
using setu::coordinator::datatypes::UseCommInstruction;
using setu::coordinator::datatypes::Program;
//==============================================================================
class Worker {
 public:
  Worker(Device device, std::size_t reply_port);
  ~Worker();

  void Start();
  void Stop();

  [[nodiscard]] bool IsRunning() const { return worker_running_.load(); }
  [[nodiscard]] const Device& GetDevice() const { return device_; }

  virtual void Execute(const Program& program) = 0;
  virtual void Setup() = 0; 

 protected:
  void InitZmqSockets();
  void CloseZmqSockets();

  void WorkerLoop();

  Device device_;
  // Zmq context and sockets
  ZmqContextPtr zmq_context_;
  ZmqSocketPtr reply_socket_;

  std::size_t reply_port_;

  std::atomic<bool> worker_running_;

  std::thread worker_thread_;
};
//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
