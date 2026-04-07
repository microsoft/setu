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
#include "commons/datatypes/TensorShardHandle.h"
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/utils/PendingOperations.h"
#include "commons/utils/RequestRouter.h"
#include "commons/utils/ThreadingUtils.h"
#include "commons/utils/ZmqHelper.h"
#include "commons/utils/ring/CompletionEntry.h"
#include "commons/utils/ring/ShmRing.h"
#include "messaging/Messages.h"
#include "node_manager/worker/Worker.h"
#include "planner/Constants.h"
#include "planner/Planner.h"
#include "planner/ir/llc/ShardAccess.h"
//==============================================================================
namespace setu::node_manager {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::Identity;
using setu::commons::NodeId;
using setu::commons::Queue;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorShardMetadataMap;
using setu::commons::datatypes::TensorShardMetadataPtr;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardsConcurrentMap;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::datatypes::TensorSpec;
using setu::commons::datatypes::TensorSpecMap;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::ClientRequest;
using setu::commons::messages::ConnectRequest;
using setu::commons::messages::ConnectResponse;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::CopyOperationFinishedRequest;
using setu::commons::messages::DeregisterShardsRequest;
using setu::commons::messages::DeregisterShardsResponse;
using setu::commons::messages::ExecuteRequest;
using setu::commons::messages::GetTensorHandleRequest;
using setu::commons::messages::GetTensorHandleResponse;
using setu::commons::messages::GetTensorSelectionRequest;
using setu::commons::messages::GetTensorSelectionResponse;
using setu::commons::messages::GetTensorSpecRequest;
using setu::commons::messages::GetTensorSpecResponse;
using setu::commons::messages::OnboardNodeAgentRequest;
using setu::commons::messages::OnboardNodeAgentResponse;
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
using setu::commons::utils::ring::CompletionEntry;
using setu::commons::utils::ring::CompletionRingProducer;
using setu::commons::utils::ring::ShmRing;
using setu::node_manager::worker::Worker;
using setu::node_manager::worker::WorkerCompletion;
using setu::node_manager::worker::WorkerTask;
using setu::planner::Plan;
using setu::planner::ir::llc::Program;
using setu::planner::ir::llc::ShardAccessMap;
using setu::planner::ir::llc::ShardAccessMode;
using setu::planner::ir::ref::BufferRef;
using setu::planner::ir::ref::RegisterRef;
using setu::planner::ir::ref::ShardRef;

using RegisterResolver = std::function<DevicePtr(const RegisterRef&)>;
//==============================================================================

/// @brief Tracks a single in-flight copy operation dispatched to workers.
struct InFlightEntry {
  std::int32_t remaining_workers;
  ShardAccessMap shard_access_map;
  std::chrono::steady_clock::time_point dispatched_at;
};

/// @brief Per-shard lock state: ref count + file lock handle.
/// When ref_count transitions 0→1, the file lock is acquired.
/// When ref_count transitions 1→0, the file lock is released.
struct ShardLockState {
  std::int32_t ref_count = 0;
  ShardAccessMode mode = ShardAccessMode::kRead;
  setu::commons::datatypes::TensorShardReadHandlePtr read_handle;
  setu::commons::datatypes::TensorShardWriteHandlePtr write_handle;
};

/// @brief Shared state between Dispatcher and Poller threads.
struct AsyncExecutorState {
  /// Per-worker input queues (Dispatcher pushes, Worker pulls).
  std::unordered_map<DeviceRank, Queue<WorkerTask>> worker_queues;

  /// Completion queue (Workers push, Poller pulls).
  Queue<WorkerCompletion> completion_queue;

  /// In-flight tracking and shard locks.
  /// Protected by in_flight_mutex. Only held briefly for counter ops,
  /// except when acquiring file locks (which may block if a client holds one).
  std::mutex in_flight_mutex;
  std::unordered_map<CopyOperationId, InFlightEntry> in_flight;

  /// Per-shard lock state: ref count + file lock handle.
  /// Dispatcher acquires locks and increments, Poller decrements and releases.
  std::unordered_map<ShardId, ShardLockState> shard_locks;

  /// Notified when any shard ref count reaches zero.
  std::condition_variable shard_ref_zero_cv;
};

//==============================================================================
class NodeAgent {
 public:
  NodeAgent(NodeId node_id, std::size_t port, std::string coordinator_endpoint,
            const std::vector<Device>& devices,
            std::string lock_base_dir = GetDefaultLockBaseDir(),
            std::string metrics_endpoint = "",
            std::size_t register_size = setu::planner::kRegisterSize);
  ~NodeAgent();

  void Start();
  void Stop();

  [[nodiscard]] std::size_t GetPort() const { return port_; }
  [[nodiscard]] std::size_t GetControlPort() const { return port_ + 1; }

  /// Returns a per-user default lock directory under the system temp path.
  [[nodiscard]] static std::string GetDefaultLockBaseDir();

 private:
  //============================================================================
  // Handler, Dispatcher, and Poller are private structs that each own a
  // component running in a separate thread. Since ZMQ sockets are not
  // thread-safe, each struct that uses ZMQ is responsible for creating its own
  // sockets from a shared ZMQ context (which is thread-safe).
  //============================================================================

  //============================================================================
  // Handler: Handles incoming messages from clients and coordinator
  //============================================================================
  struct Handler {
    Handler(NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
            std::size_t port, const std::string& coordinator_endpoint,
            Queue<std::pair<CopyOperationId, Plan>>& executor_queue,
            TensorShardsConcurrentMap& shard_id_to_tensor,
            std::string lock_base_dir,
            std::unordered_map<setu::planner::Participant,
                               setu::planner::RegisterSet>
                register_sets,
            OnboardNodeAgentRequest::P2PPairs p2p_pairs,
            AsyncExecutorState& async_executor_state);
    ~Handler();

    void Start();
    void Stop();

   private:
    void InitSockets();
    void CloseSockets();
    void Loop();

    // Control message dispatch (profiling toggle, etc.)
    void HandleControlMessage();

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
    void HandleGetTensorSelectionRequest(
        const Identity& client_identity,
        const GetTensorSelectionRequest& request);
    void HandleDeregisterShardsRequest(const Identity& client_identity,
                                       const DeregisterShardsRequest& request);
    void HandleConnectRequest(const Identity& client_identity,
                              const ConnectRequest& request);

    // Async coordinator message handlers (received on DEALER socket)
    void HandleAllocateTensorRequest(const AllocateTensorRequest& request);
    void HandleDeregisterShardsResponse(
        const DeregisterShardsResponse& response);
    void HandleCopyOperationFinishedRequest(
        const CopyOperationFinishedRequest& request);
    void HandleExecuteRequest(const ExecuteRequest& request);
    void HandleSubmitCopyResponse(const SubmitCopyResponse& response);

    void AllocateTensor(const TensorShardMetadata& shard_metadata);

    /// Wait until all in-flight worker operations on the given shards complete.
    void WaitForShardRefCountZero(const std::vector<ShardId>& shard_ids);

    NodeId node_id_;
    std::shared_ptr<zmq::context_t> zmq_context_;
    std::size_t port_;
    std::string coordinator_endpoint_;
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue_;

    ZmqSocketPtr client_socket_;
    ZmqSocketPtr sync_socket_;    // REQ socket for sync request-response
    ZmqSocketPtr async_socket_;   // DEALER socket for async send/receive
    ZmqSocketPtr control_socket_; // REP socket for control commands (port+1)

    std::thread thread_;
    std::atomic<bool> running_{false};
    bool profiling_active_ = false;  ///< Handler-thread only, no atomic needed

    // Routes coordinator responses back to the client that initiated the
    // request
    setu::commons::utils::RequestRouter request_router_;

    // Tracks pending copy operations: N clients (waiters) on 1 copy op
    // (blocker). WaiterId=Identity, BlockerId=CopyOperationId, Payload=void.
    setu::commons::utils::PendingOperations<Identity, CopyOperationId>
        pending_copies_;

    // Tracks pending shard allocations: N clients (waiters) on 1 shard
    // (blocker). WaiterId=Identity, BlockerId=ShardId, Payload=void.
    setu::commons::utils::PendingOperations<Identity, ShardId>
        pending_shard_allocs_;

    void OnboardWithCoordinator();

    TensorShardMetadataMap tensor_shard_metadata_map_;
    TensorSpecMap tensor_spec_cache_;
    TensorShardsConcurrentMap& shard_id_to_tensor_;
    std::string lock_base_dir_;  ///< Directory for file-based locks (IPC)

    /// Per-device register sets to send to coordinator during onboarding.
    std::unordered_map<setu::planner::Participant, setu::planner::RegisterSet>
        register_sets_;

    /// Directional P2P-capable device pairs to send during onboarding.
    OnboardNodeAgentRequest::P2PPairs p2p_pairs_;

    /// Tracks deregistration requests: 1 request (waiter) blocked by 1
    /// coordinator response key (blocker). WaiterId=RequestId,
    /// BlockerId=RequestId, Payload=DeregisterShardsRequest.
    setu::commons::utils::PendingOperations<RequestId, RequestId,
                                            DeregisterShardsRequest>
        pending_deregistrations_;

    // Completion ring state per client.
    // NB(Elton): Push() spins if the ring is full, which blocks the
    // Handler thread. For v0 we don't offer a more robust fallback
    // (overflow queue, dropped messages, etc.).
    static constexpr std::uint32_t kCompletionRingCapacity = 1024;

    /// Maps client identity to its completion ring producer.
    std::unordered_map<Identity, std::unique_ptr<CompletionRingProducer>>
        client_rings_;

    /// Tracks shared-memory resources per client for cleanup.
    struct ClientRingInfo {
      std::string shm_name;
      void* mmap_ptr;
      std::size_t mmap_size;
    };
    std::unordered_map<Identity, ClientRingInfo> client_ring_info_;

    /// Tracks which client + local_id to notify per copy operation.
    struct ClientLocalId {
      Identity canonical_client_id;
      std::uint64_t local_id;
    };
    std::unordered_map<CopyOperationId, std::vector<ClientLocalId>,
                       boost::hash<CopyOperationId>>
        copy_op_to_clients_;

    /// Maps request_id → local_id for in-flight submit requests.
    std::unordered_map<RequestId, std::uint64_t, boost::hash<RequestId>>
        request_id_to_local_id_;

    /// Extracts the canonical client ID by stripping the socket suffix
    /// (_query, _submit) from a ZMQ identity.
    [[nodiscard]] static Identity ResolveCanonicalClientId(
        const Identity& identity);

    AsyncExecutorState& async_executor_state_;
  };

  //============================================================================
  // Dispatcher: Embellishes programs and dispatches them to worker queues.
  // Does not wait for workers to finish — fires and moves on.
  //============================================================================
  struct Dispatcher {
    Dispatcher(NodeId node_id,
               Queue<std::pair<CopyOperationId, Plan>>& executor_queue,
               TensorShardsConcurrentMap const& shard_id_to_tensor,
               RegisterResolver register_resolver,
               AsyncExecutorState& state);
    ~Dispatcher();

    void Start();
    void Stop();

   private:
    void Loop();
    void EmbellishProgram(Program& program);

    NodeId node_id_;
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue_;
    TensorShardsConcurrentMap const& shard_id_to_tensor_;
    RegisterResolver register_resolver_;
    AsyncExecutorState& state_;

    std::thread thread_;
    std::atomic<bool> running_{false};
  };

  //============================================================================
  // Poller: Drains the completion queue and notifies the coordinator when
  // all workers for a copy operation have finished.
  //============================================================================
  struct Poller {
    Poller(NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
           const std::string& coordinator_endpoint,
           AsyncExecutorState& state);
    ~Poller();

    void Start();
    void Stop();

   private:
    void InitSockets();
    void CloseSockets();
    void Loop();

    NodeId node_id_;
    std::shared_ptr<zmq::context_t> zmq_context_;
    std::string coordinator_endpoint_;
    AsyncExecutorState& state_;

    ZmqSocketPtr async_socket_;  // DEALER socket for async send to coordinator
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

  AsyncExecutorState async_executor_state_;
  std::unique_ptr<Dispatcher> dispatcher_;
  std::unique_ptr<Poller> poller_;

  TensorShardsConcurrentMap shard_id_to_tensor_;
  std::string lock_base_dir_;  ///< Directory for file-based locks (IPC)
  std::string
      metrics_endpoint_;       ///< Telemetry server endpoint (empty = disabled)
  std::size_t register_size_;  ///< Per-register buffer size in bytes
};
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
