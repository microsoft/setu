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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
//==============================================================================
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorShard.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/messages/Messages.h"
#include "commons/utils/ThreadingUtils.h"
#include "commons/utils/ZmqHelper.h"
#include "coordinator/datatypes/CopyOperation.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::DeviceRank;
using setu::commons::Identity;
using setu::commons::NodeId;
using setu::commons::Queue;
using setu::commons::TensorName;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::WaitForCopyRequest;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
using setu::coordinator::datatypes::CopyOperationPtr;
//==============================================================================
class Coordinator {
 public:
  Coordinator(std::size_t port);

  ~Coordinator();

  std::optional<TensorShardRef> RegisterTensorShard(
      const TensorShardSpec& shard_spec);

  std::optional<CopyOperationId> SubmitCopy(const CopySpec& copy_spec);

  void PlanExecuted(CopyOperationId copy_op_id);

  void Start();
  void Stop();

 private:
  void StartHandlerLoop();
  void StopHandlerLoop();

  void StartExecutorLoop();
  void StopExecutorLoop();

  void HandlerLoop();
  void ExecutorLoop();

  void HandleNodeAgentRequest(const Identity& node_agent_identity,
                              const RegisterTensorShardRequest& request);
  void HandleNodeAgentRequest(const Identity& node_agent_identity,
                              const SubmitCopyRequest& request);
  void HandleNodeAgentRequest(const Identity& node_agent_identity,
                              const WaitForCopyRequest& request);

  void InitZmqSockets();
  void CloseZmqSockets();

  std::size_t port_;
  std::shared_ptr<zmq::context_t> zmq_context_;
  ZmqSocketPtr node_agent_socket_;

  std::thread handler_thread_;
  std::thread executor_thread_;

  std::atomic<bool> handler_running_{false};
  std::atomic<bool> executor_running_{false};
};
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
