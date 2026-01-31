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
#include "node_manager/worker/Worker.h"
#include "planner/Planner.h"
//==============================================================================
namespace setu::node_manager {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::DeviceRank;
using setu::commons::Identity;
using setu::commons::NodeId;
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
using setu::node_manager::worker::Worker;
using setu::planner::Plan;
//==============================================================================
class NodeAgent {
 public:
  NodeAgent(NodeId node_id, std::size_t port, std::string coordinator_endpoint,
            const std::vector<Device>& devices);
  ~NodeAgent();

  std::optional<TensorShardRef> RegisterTensorShard(
      const TensorShardSpec& shard_spec);

  std::optional<CopyOperationId> SubmitCopy(const CopySpec& copy_spec);

  void WaitForCopy(CopyOperationId copy_op_id);

  void CopyOperationFinished(CopyOperationId copy_op_id);

  void Execute(Plan plan);

  void Start();
  void Stop();

 private:
  //============================================================================
  // Handler and Executor are private structs that each own a component running
  // in a separate thread. Since ZMQ sockets are not thread-safe, each struct is
  // responsible for creating its own sockets from a shared ZMQ context (which
  // is thread-safe). This design prevents accidental sharing of sockets across
  // threads and keeps socket lifecycle management clean and localized.
  //============================================================================

  //============================================================================
  // Handler: Handles incoming messages from clients and coordinator
  //============================================================================
  struct Handler {
    Handler(NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
            std::size_t port, const std::string& coordinator_endpoint,
            Queue<std::pair<CopyOperationId, Plan>>& executor_queue);
    ~Handler();

    void Start();
    void Stop();

   private:
    void InitSockets();
    void CloseSockets();
    void Loop();

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

    void AllocateTensor(const TensorShardSpec& tensor_shard_spec);

    NodeId node_id_;
    std::shared_ptr<zmq::context_t> zmq_context_;
    std::size_t port_;
    std::string coordinator_endpoint_;
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue_;

    ZmqSocketPtr client_socket_;
    ZmqSocketPtr coordinator_socket_;

    std::thread thread_;
    std::atomic<bool> running_{false};

    // stores mapping from request id to the client identity who sent this
    // request. Used to route coordinator responses back to the client that
    // initiated the request
    std::unordered_map<RequestId, Identity> request_id_to_client_identity_;

    // Pending client waits: maps copy_op_id to list of client identities
    // waiting
    std::unordered_map<CopyOperationId, std::vector<Identity>,
                       boost::hash<CopyOperationId>>
        pending_waits_;

    std::unordered_map<TensorName, TensorShardSpec> tensor_name_to_spec_;
    std::unordered_map<TensorName, torch::Tensor> tensor_name_to_tensor_;
  };

  //============================================================================
  // Executor: Executes plans by dispatching to workers
  //============================================================================
  struct Executor {
    Executor(NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
             const std::string& coordinator_endpoint,
             const std::vector<Device>& devices,
             Queue<std::pair<CopyOperationId, Plan>>& executor_queue);
    ~Executor();

    void Start();
    void Stop();

   private:
    void InitSockets();
    void CloseSockets();
    void Loop();

    NodeId node_id_;
    std::shared_ptr<zmq::context_t> zmq_context_;
    std::string coordinator_endpoint_;
    std::vector<Device> devices_;
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue_;

    ZmqSocketPtr coordinator_socket_;
    std::unordered_map<DeviceRank, ZmqSocketPtr> worker_sockets_;

    std::thread thread_;
    std::atomic<bool> running_{false};
  };

  NodeId node_id_;

  std::size_t port_;
  std::string coordinator_endpoint_;
  std::vector<Device> devices_;

  std::shared_ptr<zmq::context_t> zmq_context_;

  std::unordered_map<DeviceRank, std::unique_ptr<Worker>> workers_;

  // Executor queue: (copy_op_id, node_plan) pairs for execution
  Queue<std::pair<CopyOperationId, Plan>> executor_queue_;

  std::unique_ptr<Handler> handler_;
  std::unique_ptr<Executor> executor_;
};
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
