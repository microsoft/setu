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
#include "commons/utils/PendingOperations.h"
#include "coordinator/DispatchManager.h"
#include "coordinator/Types.h"
#include "messaging/Messages.h"
#include "metastore/MetaStore.h"
#include "telemetry/MetricsSink.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::DeviceRank;
using setu::commons::NodeId;
using setu::commons::messages::DeregisterShardsRequest;
using setu::commons::messages::ExecuteResponse;
using setu::commons::messages::GetTensorSpecRequest;
using setu::commons::messages::OnboardNodeAgentRequest;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::SubmitPullRequest;
using setu::commons::utils::PendingOperations;
using setu::metastore::MetaStore;
//==============================================================================

/// @brief Processes incoming requests from the inbox queue (pure business
/// logic, no ZMQ).
///
/// Handler runs on a dedicated thread and dispatches incoming NodeAgent
/// requests to the appropriate handler method. It communicates results
/// back through the outbox queue.
class Handler {
 public:
  Handler(Queue<InboxMessage>& inbox_queue, Queue<OutboxMessage>& outbox_queue,
          MetaStore& metastore, Queue<ExecutorTask>& planner_queue,
          OutboxNotifyFn outbox_notify,
          setu::telemetry::MetricsSinkPtr metrics_sink);

  void Start();
  void Stop();

 private:
  void Loop();

  void PushOutbox(OutboxMessage msg);

  void HandleRegisterTensorShardRequest(
      const Identity& node_agent_identity,
      const RegisterTensorShardRequest& request);
  void HandleSubmitCopyRequest(const Identity& node_agent_identity,
                               const SubmitCopyRequest& request);
  void HandleSubmitPullRequest(const Identity& node_agent_identity,
                               const SubmitPullRequest& request);
  void HandleExecuteResponse(const Identity& node_identity,
                             const ExecuteResponse& response);
  void HandleGetTensorSpecRequest(const Identity& node_agent_identity,
                                  const GetTensorSpecRequest& request);
  void HandleDeregisterShardsRequest(const Identity& node_agent_identity,
                                     const DeregisterShardsRequest& request);
  void HandleOnboardNodeAgentRequest(const Identity& node_agent_identity,
                                     const OnboardNodeAgentRequest& request);

  /// @brief Unified shard submission logic for both Copy and Pull.
  void HandleShardSubmission(DispatchManager::ShardSubmission submission);

  Queue<InboxMessage>& inbox_queue_;
  Queue<OutboxMessage>& outbox_queue_;
  MetaStore& metastore_;
  Queue<ExecutorTask>& planner_queue_;
  OutboxNotifyFn outbox_notify_;
  setu::telemetry::MetricsSinkPtr metrics_sink_;

  DispatchManager dispatch_manager_;

  /// Tracks deregistration requests blocked by in-flight copy operations.
  /// WaiterId=RequestId, BlockerId=CopyOperationId,
  /// Payload=PendingDeregistration. As each copy completes, Resolve() is
  /// called and the deregistration payload is returned when all its
  /// blockers are resolved.
  PendingOperations<RequestId, CopyOperationId, PendingDeregistration>
      deregistration_tracker_;

  std::thread thread_;
  std::atomic<bool> running_{false};
};
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
