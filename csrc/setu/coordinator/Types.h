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
#include "messaging/Messages.h"
#include "planner/Participant.h"
#include "planner/RegisterSet.h"
#include "planner/hints/HintStore.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::Identity;
using setu::commons::Queue;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::CopySpec;
using setu::planner::hints::HintStore;
//==============================================================================
// Message type aliases
//==============================================================================
using CoordinatorMessage = setu::commons::messages::CoordinatorMessage;
using NodeAgentRequest = setu::commons::messages::NodeAgentRequest;
//==============================================================================
// Callback type for outbox wake-up notification
//==============================================================================
using OutboxNotifyFn = std::function<void()>;
//==============================================================================
// Message types for internal queues
//==============================================================================
struct InboxMessage {
  Identity node_agent_identity;
  NodeAgentRequest request;
};

struct OutboxMessage {
  Identity node_agent_identity;
  CoordinatorMessage message;
};
//==============================================================================
// Shared cross-thread state for tracking a copy operation
//==============================================================================

/// @brief Shared state for tracking a copy operation across Handler and
/// Executor threads.
///
/// Thread Safety: expected_responses is std::atomic because Executor writes
/// it (after dispatching ExecuteRequests) and Handler reads it (when
/// processing ExecuteResponses). These accesses occur without explicit queue
/// synchronization for this field, so we use release/acquire ordering to
/// ensure visibility.
struct CopyOperationState {
  CopySpec spec;
  std::vector<Identity> submitters;  // NodeAgents to notify when done

  // Atomic: Executor writes (release), Handler reads (acquire).
  std::atomic<std::size_t> expected_responses{0};

  std::size_t completed_responses{0};  // Handler-thread only (not shared)

  /// @brief Timestamp when the copy operation was first submitted.
  std::chrono::high_resolution_clock::time_point start_time;

  explicit CopyOperationState(CopySpec spec_param,
                              std::vector<Identity> submitters_param)
      : spec(std::move(spec_param)), submitters(std::move(submitters_param)) {}
};
using CopyOperationStatePtr = std::shared_ptr<CopyOperationState>;
//==============================================================================
// Deregistration payload
//==============================================================================

/// @brief Payload for a deregistration request deferred until all blocking
/// copy operations complete.
struct PendingDeregistration {
  Identity node_agent_identity;
  RequestId request_id;
  std::unordered_map<TensorName, std::vector<ShardId>> shards_by_tensor;
};
//==============================================================================
// Planner task
//==============================================================================

/// @brief Task for the planner containing CopyOperationId, CopySpec,
/// shared state, and per-operation hints from the first shard submission.
struct PlannerTask {
  CopyOperationId copy_op_id;
  CopySpec copy_spec;
  CopyOperationStatePtr state;  // Shared with Handler's copy_operations_ map
  HintStore hints;              // Per-operation hints (first-writer-wins)
  std::optional<std::vector<std::string>> pass_names;  // first-writer-wins
};
//==============================================================================
// Onboarding task
//==============================================================================

/// @brief Task to add register sets and P2P topology to the planner backend.
struct OnboardingTask {
  Identity node_agent_identity;
  RequestId request_id;
  std::unordered_map<setu::planner::Participant, setu::planner::RegisterSet>
      register_sets;
  std::vector<setu::commons::messages::P2PPair> p2p_pairs;
};

/// @brief Variant of tasks the Executor can process.
using ExecutorTask = std::variant<PlannerTask, OnboardingTask>;
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
