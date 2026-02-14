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
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/utils/PendingOperations.h"
#include "commons/utils/RequestRouter.h"
#include "commons/utils/ThreadingUtils.h"
#include "commons/utils/ZmqHelper.h"
#include "messaging/Messages.h"
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
using setu::commons::TensorShardsConcurrentMap;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorShardMetadataMap;
using setu::commons::datatypes::TensorShardMetadataPtr;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::ClientRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::CopyOperationFinishedRequest;
using setu::commons::messages::ExecuteRequest;
using setu::commons::messages::GetTensorHandleRequest;
using setu::commons::messages::GetTensorHandleResponse;
using setu::commons::messages::RegisterTensorShardCoordinatorResponse;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::SubmitPullRequest;
using setu::commons::messages::WaitForCopyRequest;
using setu::commons::messages::WaitForCopyResponse;
using setu::commons::messages::WaitForShardAllocationRequest;
using setu::commons::messages::WaitForShardAllocationResponse;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
using setu::ir::Program;
using setu::ir::ShardRef;
using setu::node_manager::worker::Worker;
using setu::planner::Plan;
//==============================================================================
class NodeAgent {
 public:
  NodeAgent(NodeId node_id, std::size_t port, std::string coordinator_endpoint,
            const std::vector<Device>& devices,
            std::string lock_base_dir = "/tmp/setu/locks");
  ~NodeAgent();

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
            Queue<std::pair<CopyOperationId, Plan>>& executor_queue,
            TensorShardsConcurrentMap& shard_id_to_tensor,
            std::string lock_base_dir);
    ~Handler();

    void Start();
    void Stop();

   private:
    void InitSockets();
    void CloseSockets();
    void Loop();

    // Client message dispatch
    void HandleClientMessage(const Identity& client_identity,
                             const ClientRequest& request);

    // Async coordinator message dispatch (messages received on DEALER socket)
    void HandleAsyncCoordinatorMessage(const CoordinatorMessage& message);

    // Client message handlers
    // Sync: RegisterTensorShard sends via REQ socket, blocks for response
    void HandleRegisterTensorShardRequest(
        const Identity& client_identity,
        const RegisterTensorShardRequest& request);
    // Async: SubmitCopy/SubmitPull send via DEALER socket, response comes later
    void HandleSubmitCopyRequest(const Identity& client_identity,
                                 const SubmitCopyRequest& request);
    void HandleSubmitPullRequest(const Identity& client_identity,
                                 const SubmitPullRequest& request);
    // Local: handled entirely within NodeAgent
    void HandleWaitForCopyRequest(const Identity& client_identity,
                                  const WaitForCopyRequest& request);
    void HandleGetTensorHandleRequest(const Identity& client_identity,
                                      const GetTensorHandleRequest& request);
    void HandleWaitForShardAllocationRequest(
        const Identity& client_identity,
        const WaitForShardAllocationRequest& request);

    // Async coordinator message handlers (received on DEALER socket)
    void HandleAllocateTensorRequest(const AllocateTensorRequest& request);
    void HandleCopyOperationFinishedRequest(
        const CopyOperationFinishedRequest& request);
    void HandleExecuteRequest(const ExecuteRequest& request);
    void HandleSubmitCopyResponse(const SubmitCopyResponse& response);

    void AllocateTensor(const TensorShardMetadata& shard_metadata);

    NodeId node_id_;
    std::shared_ptr<zmq::context_t> zmq_context_;
    std::size_t port_;
    std::string coordinator_endpoint_;
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue_;

    ZmqSocketPtr client_socket_;
    ZmqSocketPtr sync_socket_;   // REQ socket for sync request-response
    ZmqSocketPtr async_socket_;  // DEALER socket for async send/receive

    std::thread thread_;
    std::atomic<bool> running_{false};

    // Routes coordinator responses back to the client that initiated the
    // request
    setu::commons::utils::RequestRouter request_router_;

    // Tracks pending copy operations: registration, waiting, and completion
    setu::commons::utils::PendingOperations<CopyOperationId> pending_copies_;

    // Tracks pending shard allocation: registration, waiting, and completion
    setu::commons::utils::PendingOperations<ShardId> pending_shard_allocs_;

    TensorShardMetadataMap tensor_shard_metadata_map_;
    TensorShardsConcurrentMap& shard_id_to_tensor_;
    std::string lock_base_dir_;  ///< Directory for file-based locks (IPC)
  };

  //============================================================================
  // Executor: Executes plans by dispatching to workers
  //============================================================================
  struct Executor {
    Executor(NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
             const std::string& coordinator_endpoint,
             const std::vector<Device>& devices,
             Queue<std::pair<CopyOperationId, Plan>>& executor_queue,
             TensorShardsConcurrentMap const& shard_id_to_tensor);
    ~Executor();

    void Start();
    void Stop();

   private:
    void InitSockets();
    void CloseSockets();
    void Loop();
    void EmbellishProgram(Program& program);

    NodeId node_id_;
    std::shared_ptr<zmq::context_t> zmq_context_;
    std::string coordinator_endpoint_;
    std::vector<Device> devices_;
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue_;

    ZmqSocketPtr async_socket_;  // DEALER socket for async send to coordinator
    std::unordered_map<DeviceRank, ZmqSocketPtr> worker_sockets_;

    std::thread thread_;
    std::atomic<bool> running_{false};
    TensorShardsConcurrentMap const& shard_id_to_tensor_;
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

  TensorShardsConcurrentMap shard_id_to_tensor_;
  std::string lock_base_dir_;  ///< Directory for file-based locks (IPC)
};
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
