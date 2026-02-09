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
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "messaging/Messages.h"
#include "commons/utils/ShardAggregator.h"
#include "commons/utils/ThreadingUtils.h"
#include "commons/utils/ZmqHelper.h"
#include "coordinator/datatypes/CopyOperation.h"
#include "metastore/MetaStore.h"
#include "planner/backends/nccl.h"
//==============================================================================
namespace setu::coordinator {
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
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::SubmitPullRequest;
using setu::commons::messages::WaitForCopyRequest;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
using setu::coordinator::datatypes::CopyOperationPtr;
using setu::metastore::MetaStore;
using setu::planner::backends::nccl::NCCLPlanner;
using setu::planner::Plan;

/// @brief Shared state for tracking a copy operation across Handler and
/// Executor threads.
///
/// Thread Safety: expected_responses is std::atomic because Executor writes it
/// (after dispatching ExecuteRequests) and Handler reads it (when processing
/// ExecuteResponses). These accesses occur without explicit queue
/// synchronization for this field, so we use release/acquire ordering to ensure
/// visibility.
struct CopyOperationState {
  CopySpec spec;
  std::vector<Identity> submitters;  // NodeAgents to notify when done

  // Atomic because Executor writes and Handler reads without explicit
  // synchronization. Using release/acquire ordering ensures Executor's write
  // is visible to Handler.
  std::atomic<std::size_t> expected_responses{0};

  std::size_t completed_responses{0};  // Only Handler reads/writes

  explicit CopyOperationState(CopySpec spec_param,
                              std::vector<Identity> submitters_param)
      : spec(std::move(spec_param)), submitters(std::move(submitters_param)) {}
};
using CopyOperationStatePtr = std::shared_ptr<CopyOperationState>;

/// @brief Task for the planner containing CopyOperationId, CopySpec, and shared
/// state
struct PlannerTask {
  CopyOperationId copy_op_id;
  CopySpec copy_spec;
  CopyOperationStatePtr state;  // Shared with Handler's copy_operations_ map
};

//==============================================================================
class Coordinator {
 public:
  Coordinator(std::size_t port);

  ~Coordinator();

  std::optional<TensorShardMetadata> RegisterTensorShard(
      const TensorShardSpec& shard_spec);

  std::optional<CopyOperationId> SubmitCopy(const CopySpec& copy_spec);

  void PlanExecuted(CopyOperationId copy_op_id);

  void Start();
  void Stop();

 private:
  MetaStore metastore_;
  //============================================================================
  // Architecture: Gateway + Queues
  //
  // Both the Handler and Executor have to send messages back to the NodeAgent
  // through a Router socket. But, ZMQ sockets are not thread-safe. So, we
  // would need to bind two ZMQ sockets on different ports, because we cannot
  // bind two sockets to the same port using the same shared context. To
  // workaround this complexity, we use a single I/O thread (Gateway) that owns
  // the ZMQ socket. Internal components (Handler, Executor) communicate with
  // Gateway through thread-safe queues:
  //
  //   NodeAgents <---> [Gateway] <---> inbox_queue_  ---> Handler
  //                        ^
  //                        |
  //                    outbox_queue_ <--- Handler / Executor
  //
  // This keeps ZMQ isolated to one thread and makes Handler/Executor pure
  // business logic with no networking concerns.
  //============================================================================

  //============================================================================
  // Message types for internal queues
  //============================================================================
  using CoordinatorMessage = setu::commons::messages::CoordinatorMessage;
  using NodeAgentRequest = setu::commons::messages::NodeAgentRequest;

  struct InboxMessage {
    Identity node_agent_identity;
    NodeAgentRequest request;
  };

  struct OutboxMessage {
    Identity node_agent_identity;
    CoordinatorMessage message;
  };

  //============================================================================
  // Gateway: Owns ZMQ socket, handles all network I/O
  //============================================================================
  struct Gateway {
    Gateway(std::shared_ptr<zmq::context_t> zmq_context, std::size_t port,
            Queue<InboxMessage>& inbox_queue,
            Queue<OutboxMessage>& outbox_queue);
    ~Gateway();

    void Start();
    void Stop();

   private:
    void InitSockets();
    void CloseSockets();
    void Loop();

    std::shared_ptr<zmq::context_t> zmq_context_;
    std::size_t port_;

    Queue<InboxMessage>& inbox_queue_;
    Queue<OutboxMessage>& outbox_queue_;

    ZmqSocketPtr node_agent_socket_;

    std::thread thread_;
    std::atomic<bool> running_{false};
  };

  //============================================================================
  // Handler: Processes incoming requests (pure business logic, no ZMQ)
  //============================================================================
  struct Handler {
    Handler(Queue<InboxMessage>& inbox_queue,
            Queue<OutboxMessage>& outbox_queue, MetaStore& metastore,
            Queue<PlannerTask>& planner_queue);

    void Start();
    void Stop();

   private:
    void Loop();

    void HandleRegisterTensorShardRequest(
        const Identity& node_agent_identity,
        const RegisterTensorShardRequest& request);
    void HandleSubmitCopyRequest(const Identity& node_agent_identity,
                                 const SubmitCopyRequest& request);
    void HandleSubmitPullRequest(const Identity& node_agent_identity,
                                 const SubmitPullRequest& request);
    void HandleExecuteResponse(const Identity& node_identity,
                               const setu::commons::messages::ExecuteResponse&
                                   response);

    /// @brief Unified shard submission logic for both Copy and Pull.
    void HandleShardSubmission(const Identity& node_agent_identity,
                               const RequestId& request_id,
                               const ShardId& shard_id,
                               const CopySpec& copy_spec,
                               std::size_t expected_shards);

    /// Key for tracking copy operations by (src, dst) tensor pair
    struct CopyKey {
      TensorName src_name;
      TensorName dst_name;

      bool operator<(const CopyKey& other) const {
        if (src_name != other.src_name) return src_name < other.src_name;
        return dst_name < other.dst_name;
      }
    };

    Queue<InboxMessage>& inbox_queue_;
    Queue<OutboxMessage>& outbox_queue_;
    MetaStore& metastore_;
    Queue<PlannerTask>& planner_queue_;

    /// Aggregates shard submissions per (src, dst) pair until all expected
    /// shards arrive
    /// TODO: we might have copies with distinct TensorSelections, we need to
    /// address that
    setu::commons::utils::ShardAggregator<CopyKey, CopySpec> shard_aggregator_;

    /// Maps CopyOperationId to shared CopyOperationState (includes submitters
    /// and completion tracking)
    std::map<CopyOperationId, CopyOperationStatePtr> copy_operations_;

    std::thread thread_;
    std::atomic<bool> running_{false};
  };

  //============================================================================
  // Executor: Compiles CopySpecs and dispatches execution plans to NodeAgents
  //============================================================================
  struct Executor {
    Executor(Queue<PlannerTask>& planner_queue, Queue<OutboxMessage>& outbox_queue,
             MetaStore& metastore);

    void Start();
    void Stop();

   private:
    void Loop();

    Queue<PlannerTask>& planner_queue_;
    Queue<OutboxMessage>& outbox_queue_;
    MetaStore& metastore_;
    NCCLPlanner planner_;

    std::thread thread_;
    std::atomic<bool> running_{false};
  };

  std::size_t port_;

  std::shared_ptr<zmq::context_t> zmq_context_;

  // Internal message queues
  Queue<InboxMessage> inbox_queue_;
  Queue<OutboxMessage> outbox_queue_;

  /// Queue of PlannerTasks (CopyOperationId + CopySpec) for the Executor to compile and dispatch
  Queue<PlannerTask> planner_queue_;

  std::unique_ptr<Gateway> gateway_;
  std::unique_ptr<Handler> handler_;
  std::unique_ptr<Executor> executor_;
};
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
