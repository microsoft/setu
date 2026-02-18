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
#include "commons/datatypes/Device.h"
#include "commons/enums/Enums.h"
#include "commons/utils/ZmqHelper.h"
#include "planner/ir/llc/Instruction.h"
#include "planner/ir/ref/RegisterRef.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::NodeId;
using setu::commons::datatypes::Device;
using setu::commons::enums::ErrorCode;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
using setu::planner::ir::llc::Program;
using setu::planner::ir::ref::RegisterRef;
//==============================================================================
class Worker {
 public:
  Worker(NodeId node_id, Device device);
  ~Worker();

  void Connect(ZmqContextPtr zmq_context, std::string endpoint);

  void Start();
  void Stop();

  [[nodiscard]] bool IsRunning() const { return worker_running_.load(); }
  [[nodiscard]] const Device& GetDevice() const { return device_; }
  [[nodiscard]] const std::string& GetEndpoint() const { return endpoint_; }

  virtual void Execute(const Program& program) = 0;
  virtual void Setup() = 0;

  /// Resolve a register reference to its device pointer.
  [[nodiscard]] virtual DevicePtr ResolveRegister(
      const RegisterRef& ref) const = 0;

 protected:
  void InitZmqSockets();
  void CloseZmqSockets();

  void WorkerLoop();

  NodeId node_id_;
  Device device_;

  std::string endpoint_;
  ZmqContextPtr zmq_context_;
  ZmqSocketPtr socket_;

  std::atomic<bool> worker_running_;

  std::thread worker_thread_;
};
//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
