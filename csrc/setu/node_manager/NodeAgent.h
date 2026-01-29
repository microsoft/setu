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
#include "commons/TorchCommon.h"
#include "commons/Types.h"
//==============================================================================
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorShard.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/messages/Messages.h"
#include "commons/utils/ThreadingUtils.h"
#include "commons/utils/ZmqHelper.h"
#include "coordinator/datatypes/Plan.h"
#include "node_manager/worker/Worker.h"
//==============================================================================
namespace setu::node_manager {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::DeviceRank;
using setu::commons::Identity;
using setu::commons::NodeRank;
using setu::commons::Queue;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::ClientRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::CopyOperationFinishedRequest;
using setu::commons::messages::ExecuteRequest;
using setu::commons::messages::GetTensorHandleRequest;
using setu::commons::messages::GetTensorHandleResponse;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::RegisterTensorShardResponse;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::WaitForCopyRequest;
using setu::commons::messages::WaitForCopyResponse;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
using setu::coordinator::datatypes::Plan;
using setu::node_manager::worker::Worker;
//==============================================================================
class NodeAgent {
 public:
  NodeAgent(NodeRank node_rank, std::size_t router_port,
            std::size_t dealer_executor_port, std::size_t dealer_handler_port,
            const std::vector<Device>& devices);
  ~NodeAgent();

  std::optional<TensorShardRef> RegisterTensorShard(
      const TensorShardSpec& shard_spec);

  std::optional<CopyOperationId> SubmitCopy(const CopySpec& copy_spec);

  void WaitForCopy(CopyOperationId copy_op_id);

  void AllocateTensor(const TensorShardSpec& tensor_shard_spec);

  void CopyOperationFinished(CopyOperationId copy_op_id);

  void Execute(Plan plan);

  void Start();
  void Stop();

 private:
  void StartHandlerLoop();
  void StopHandlerLoop();

  void StartExecutorLoop();
  void StopExecutorLoop();

  void HandlerLoop();
  void ExecutorLoop();

  // Unified message dispatch methods
  void HandleClientMessage(const Identity& client_identity,
                           const ClientRequest& request);
  void HandleCoordinatorMessage(const CoordinatorMessage& message);

  // Client message handlers
  void HandleRegisterTensorShardRequest(
      const Identity& client_identity,
      const RegisterTensorShardRequest& request);
  void HandleSubmitCopyRequest(const Identity& client_identity,
                               const SubmitCopyRequest& request);
  void HandleWaitForCopyRequest(const Identity& client_identity,
                                const WaitForCopyRequest& request);
  void HandleGetTensorHandleRequest(const Identity& client_identity,
                                    const GetTensorHandleRequest& request);

  // Coordinator message handlers
  void HandleAllocateTensorRequest(const AllocateTensorRequest& request);
  void HandleCopyOperationFinishedRequest(
      const CopyOperationFinishedRequest& request);
  void HandleExecuteRequest(const ExecuteRequest& request);
  void HandleRegisterTensorShardResponse(
      const RegisterTensorShardResponse& response);
  void HandleSubmitCopyResponse(const SubmitCopyResponse& response);
  void HandleWaitForCopyResponse(const WaitForCopyResponse& response);

  void InitZmqSockets();
  void CloseZmqSockets();

  void InitWorkers(const std::vector<Device>& devices);

  void EnsureWorkerIsReady(DeviceRank device_rank);

  Device CreateDeviceForRank(DeviceRank device_rank) const;

  NodeRank node_rank_;

  std::shared_ptr<zmq::context_t> zmq_context_;
  ZmqSocketPtr client_router_socket_;
  ZmqSocketPtr coordinator_dealer_executor_socket_;
  ZmqSocketPtr coordinator_dealer_handler_socket_;
  std::unordered_map<DeviceRank, ZmqSocketPtr> workers_req_sockets_;

  std::unordered_map<RequestId, Identity> request_to_client_;

  std::thread handler_thread_;
  std::thread executor_thread_;

  std::size_t router_port_;
  std::size_t dealer_executor_port_;
  std::size_t dealer_handler_port_;

  std::atomic<bool> handler_running_{false};
  std::atomic<bool> executor_running_{false};

  std::unordered_map<DeviceRank, std::unique_ptr<Worker>> workers_;

  // Pending client waits: maps copy_op_id to list of client identities waiting
  std::unordered_map<CopyOperationId, std::vector<Identity>,
                     boost::hash<CopyOperationId>>
      pending_waits_;

  // Executor queue: (copy_op_id, node_plan) pairs for execution
  Queue<std::pair<CopyOperationId, Plan>> executor_queue_;

  std::unordered_map<TensorName, TensorShardSpec> tensor_name_to_spec_;
  std::unordered_map<TensorName, torch::Tensor> tensor_name_to_tensor_;
};
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
