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
#include "coordinator/Coordinator.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/QueueUtils.h"
#include "commons/utils/Comm.h"
#include "planner/targets/nccl.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::NodeId;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::StringToUUID;
using setu::commons::datatypes::TensorDim;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::CopyOperationFinishedRequest;
using setu::commons::messages::ExecuteRequest;
using setu::commons::messages::ExecuteResponse;
using setu::commons::messages::NodeAgentRequest;
using setu::commons::messages::RegisterTensorShardCoordinatorResponse;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::SubmitPullRequest;
using setu::commons::utils::Comm;
using setu::commons::utils::ZmqHelper;
using setu::planner::Plan;
//==============================================================================
constexpr std::int32_t kPollTimeoutMs = 100;
//==============================================================================
// Coordinator Implementation
//==============================================================================
Coordinator::Coordinator(std::size_t port)
    : port_(port), zmq_context_(std::make_shared<zmq::context_t>()) {
  gateway_ = std::make_unique<Gateway>(zmq_context_, port_, inbox_queue_,
                                       outbox_queue_);
  handler_ = std::make_unique<Handler>(inbox_queue_, outbox_queue_, metastore_,
                                       planner_queue_);
  executor_ =
      std::make_unique<Executor>(planner_queue_, outbox_queue_, metastore_);
}

Coordinator::~Coordinator() {
  Stop();
  if (zmq_context_) {
    zmq_context_->close();
  }
}

void Coordinator::Start() {
  LOG_DEBUG("Starting Coordinator");
  gateway_->Start();
  handler_->Start();
  executor_->Start();
}

void Coordinator::Stop() {
  LOG_DEBUG("Stopping Coordinator");

  inbox_queue_.close();
  planner_queue_.close();
  outbox_queue_.close();

  gateway_->Stop();
  handler_->Stop();
  executor_->Stop();
}

std::optional<TensorShardMetadata> Coordinator::RegisterTensorShard(
    const TensorShardSpec& shard_spec) {
  LOG_DEBUG("Registering tensor shard: {}", shard_spec.name);

  // TODO: Implement tensor shard registration
  return std::nullopt;
}

std::optional<CopyOperationId> Coordinator::SubmitCopy(
    const CopySpec& copy_spec) {
  LOG_DEBUG("Submitting copy operation from {} to {}", copy_spec.src_name,
            copy_spec.dst_name);

  // TODO: Implement copy submission and plan generation
  return std::nullopt;
}

void Coordinator::PlanExecuted(CopyOperationId copy_op_id) {
  LOG_DEBUG("Plan executed for copy operation ID: {}", copy_op_id);

  // TODO: Implement plan execution completion handling
}

//==============================================================================
// Gateway Implementation
//==============================================================================
Coordinator::Gateway::Gateway(std::shared_ptr<zmq::context_t> zmq_context,
                              std::size_t port,
                              Queue<InboxMessage>& inbox_queue,
                              Queue<OutboxMessage>& outbox_queue)
    : zmq_context_(zmq_context),
      port_(port),
      inbox_queue_(inbox_queue),
      outbox_queue_(outbox_queue) {
  InitSockets();
}

Coordinator::Gateway::~Gateway() {
  Stop();
  CloseSockets();
}

void Coordinator::Gateway::InitSockets() {
  node_agent_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, port_);
}

void Coordinator::Gateway::CloseSockets() {
  if (node_agent_socket_) {
    node_agent_socket_->close();
  }
}

void Coordinator::Gateway::Start() {
  if (running_.load()) {
    return;
  }
  thread_ = std::thread(SETU_LAUNCH_THREAD([this]() { this->Loop(); },
                                           "CoordinatorGatewayThread"));
}

void Coordinator::Gateway::Stop() {
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
}

void Coordinator::Gateway::Loop() {
  running_ = true;
  while (running_) {
    // Poll for incoming messages from NodeAgents
    auto ready = Comm::PollForRead({node_agent_socket_}, kPollTimeoutMs);

    for (const auto& socket : ready) {
      if (socket == node_agent_socket_) {
        auto [node_agent_identity, request] =
            Comm::RecvWithIdentity<NodeAgentRequest>(socket);
        auto status =
            inbox_queue_.try_push(InboxMessage{node_agent_identity, request});
        if (status == boost::queue_op_status::closed) {
          return;
        }
      }
    }

    // Send any outgoing messages (drain all available without blocking)
    try {
      while (!outbox_queue_.empty()) {
        OutboxMessage outbox_msg = outbox_queue_.pull();
        Comm::Send<CoordinatorMessage>(node_agent_socket_,
                                       outbox_msg.node_agent_identity,
                                       outbox_msg.message);
      }
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}

//==============================================================================
// Handler Implementation
//==============================================================================
Coordinator::Handler::Handler(Queue<InboxMessage>& inbox_queue,
                              Queue<OutboxMessage>& outbox_queue,
                              MetaStore& metastore,
                              Queue<PlannerTask>& planner_queue)
    : inbox_queue_(inbox_queue),
      outbox_queue_(outbox_queue),
      metastore_(metastore),
      planner_queue_(planner_queue) {}

void Coordinator::Handler::Start() {
  if (running_.load()) {
    return;
  }
  thread_ = std::thread(SETU_LAUNCH_THREAD([this]() { this->Loop(); },
                                           "CoordinatorHandlerThread"));
}

void Coordinator::Handler::Stop() {
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
}

void Coordinator::Handler::Loop() {
  running_ = true;
  while (running_) {
    try {
      InboxMessage inbox_msg = inbox_queue_.pull();
      std::visit(
          [&](const auto& msg) {
            using T = std::decay_t<decltype(msg)>;
            if constexpr (std::is_same_v<T, RegisterTensorShardRequest>) {
              HandleRegisterTensorShardRequest(inbox_msg.node_agent_identity,
                                               msg);
            } else if constexpr (std::is_same_v<T, SubmitCopyRequest>) {
              HandleSubmitCopyRequest(inbox_msg.node_agent_identity, msg);
            } else if constexpr (std::is_same_v<T, SubmitPullRequest>) {
              HandleSubmitPullRequest(inbox_msg.node_agent_identity, msg);
            } else if constexpr (std::is_same_v<T, ExecuteResponse>) {
              HandleExecuteResponse(inbox_msg.node_agent_identity, msg);
            } else {
              LOG_WARNING("Handler: Unknown message type (index={})",
                          inbox_msg.request.index());
            }
          },
          inbox_msg.request);
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}

void Coordinator::Handler::HandleRegisterTensorShardRequest(
    const Identity& node_agent_identity,
    const RegisterTensorShardRequest& request) {
  LOG_INFO("Coordinator received RegisterTensorShardRequest for tensor: {}",
           request.tensor_shard_spec.name);

  // Parse NodeId from the identity (NodeAgent REQ identity is
  // "uuid_req")
  auto underscore_pos = node_agent_identity.rfind('_');
  ASSERT_VALID_RUNTIME(underscore_pos != std::string::npos,
                       "Invalid node agent identity format: {}",
                       node_agent_identity);
  NodeId owner_node_id =
      StringToUUID(node_agent_identity.substr(0, underscore_pos));

  // Register the tensor shard in the metastore with owner information
  auto shard_metadata_ptr =
      metastore_.RegisterTensorShard(request.tensor_shard_spec, owner_node_id);

  // Send response with TensorShardMetadata
  if (shard_metadata_ptr) {
    RegisterTensorShardCoordinatorResponse response(
        request.request_id, ErrorCode::kSuccess, *shard_metadata_ptr);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
  } else {
    LOG_ERROR("Failed to register tensor shard: {}", request.tensor_shard_spec);
    RegisterTensorShardCoordinatorResponse response(
        request.request_id, ErrorCode::kInvalidArguments);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

  // Check if all shards for this tensor are registered
  if (metastore_.AllShardsRegistered(request.tensor_shard_spec.name)) {
    LOG_INFO(
        "All shards registered for tensor: {}, sending AllocateTensorRequest "
        "to all owners",
        request.tensor_shard_spec.name);

    // Get tensor metadata to find all owner NodeIds
    auto metadata =
        metastore_.GetTensorMetadata(request.tensor_shard_spec.name);
    ASSERT_VALID_POINTER_ARGUMENT(metadata);

    // Group shard IDs by owner node
    std::unordered_map<NodeId, std::vector<ShardId>> owner_to_shard_ids;
    for (const auto& [shard_id, shard_metadata] : metadata->shards) {
      owner_to_shard_ids[shard_metadata->owner].push_back(shard_id);
    }

    // Send AllocateTensorRequest to each NodeAgent's async (DEALER) socket
    for (const auto& [owner_id, shard_ids] : owner_to_shard_ids) {
      Identity owner_identity = to_string(owner_id) + "_dealer";
      AllocateTensorRequest allocate_request(shard_ids);
      outbox_queue_.push(OutboxMessage{owner_identity, allocate_request});
    }
  }
}

void Coordinator::Handler::HandleSubmitCopyRequest(
    const Identity& node_agent_identity, const SubmitCopyRequest& request) {
  LOG_INFO("Coordinator received SubmitCopyRequest from {} to {} for shard {}",
           request.copy_spec.src_name, request.copy_spec.dst_name,
           request.shard_id);

  // Expected = all src shards + all dst shards
  std::size_t expected_shards =
      metastore_.GetNumShardsForTensor(request.copy_spec.src_name) +
      metastore_.GetNumShardsForTensor(request.copy_spec.dst_name);

  HandleShardSubmission(node_agent_identity, request.request_id,
                        request.shard_id, request.copy_spec, expected_shards);
}

void Coordinator::Handler::HandleSubmitPullRequest(
    const Identity& node_agent_identity, const SubmitPullRequest& request) {
  LOG_INFO("Coordinator received SubmitPullRequest from {} to {} for shard {}",
           request.copy_spec.src_name, request.copy_spec.dst_name,
           request.shard_id);

  // For Pull: expected shards = number of DESTINATION shards only (one-sided)
  std::size_t expected_shards =
      metastore_.GetNumShardsForTensor(request.copy_spec.dst_name);

  HandleShardSubmission(node_agent_identity, request.request_id,
                        request.shard_id, request.copy_spec, expected_shards);
}

void Coordinator::Handler::HandleShardSubmission(
    const Identity& node_agent_identity, const RequestId& request_id,
    const ShardId& shard_id, const CopySpec& copy_spec,
    std::size_t expected_shards) {
  using setu::commons::utils::AggregationParticipant;

  CopyKey copy_key{copy_spec.src_name, copy_spec.dst_name};

  auto result = shard_aggregator_.Submit(
      copy_key, shard_id, copy_spec,
      AggregationParticipant{node_agent_identity, request_id}, expected_shards,
      [](const CopySpec& stored, const CopySpec& incoming) {
        /// TODO: need to handle errors differently
        ASSERT_VALID_RUNTIME(
            *incoming.src_selection == *stored.src_selection,
            "Shard submission {} -> {}: source selection mismatch",
            incoming.src_name, incoming.dst_name);
        ASSERT_VALID_RUNTIME(
            *incoming.dst_selection == *stored.dst_selection,
            "Shard submission {} -> {}: destination selection mismatch",
            incoming.src_name, incoming.dst_name);
      });

  if (!result.has_value()) {
    return;
  }

  // All shards submitted — generate CopyOperationId and dispatch
  CopyOperationId copy_op_id = GenerateUUID();

  LOG_INFO(
      "All shards submitted for {} -> {}, "
      "copy_op_id={}, adding to planner queue",
      copy_spec.src_name, copy_spec.dst_name, copy_op_id);

  // Collect submitter identities
  std::vector<Identity> submitters;
  submitters.reserve(result->participants.size());
  for (const auto& participant : result->participants) {
    submitters.push_back(participant.identity);
  }

  // Create shared state with submitter identities
  auto state = std::make_shared<CopyOperationState>(result->payload,
                                                    std::move(submitters));

  // Store the shared state (will be accessed by HandleExecuteResponse)
  copy_operations_.emplace(copy_op_id, state);

  // Add to planner queue with copy_op_id and shared state
  planner_queue_.push(PlannerTask{copy_op_id, result->payload, state});

  // Send responses to all waiting participants with copy_op_id
  for (const auto& participant : result->participants) {
    SubmitCopyResponse response(participant.request_id, copy_op_id,
                                ErrorCode::kSuccess);
    outbox_queue_.push(OutboxMessage{participant.identity, response});
  }
}

void Coordinator::Handler::HandleExecuteResponse(
    const Identity& /*node_identity*/, const ExecuteResponse& response) {
  auto it = copy_operations_.find(response.copy_op_id);
  if (it == copy_operations_.end()) {
    LOG_WARNING("ExecuteResponse for unknown copy_op_id: {}, ignoring",
                response.copy_op_id);
    return;
  }

  auto& state = it->second;
  state->completed_responses++;

  // Atomic load with acquire ordering to synchronize with Executor's write
  const auto expected =
      state->expected_responses.load(std::memory_order_acquire);

  LOG_DEBUG("ExecuteResponse for copy_op_id {}: {}/{} responses received",
            response.copy_op_id, state->completed_responses, expected);

  // Check if all participants completed
  if (state->completed_responses == expected) {
    LOG_INFO(
        "All {} participants completed for copy_op_id {}, notifying {} "
        "submitters",
        expected, response.copy_op_id, state->submitters.size());

    // Send CopyOperationFinishedRequest to all SUBMITTERS
    for (const auto& submitter_identity : state->submitters) {
      CopyOperationFinishedRequest finish_req(response.copy_op_id);
      outbox_queue_.push(OutboxMessage{submitter_identity, finish_req});
    }

    copy_operations_.erase(it);
  }
}

//==============================================================================
// Executor Implementation
//==============================================================================
Coordinator::Executor::Executor(Queue<PlannerTask>& planner_queue,
                                Queue<OutboxMessage>& outbox_queue,
                                MetaStore& metastore)
    : planner_queue_(planner_queue),
      outbox_queue_(outbox_queue),
      metastore_(metastore),
      planner_(std::make_unique<setu::planner::targets::NCCL>()) {}

void Coordinator::Executor::Start() {
  if (running_.load()) {
    return;
  }
  thread_ = std::thread(SETU_LAUNCH_THREAD([this]() { this->Loop(); },
                                           "CoordinatorExecutorThread"));
}

void Coordinator::Executor::Stop() {
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
}

void Coordinator::Executor::Loop() {
  running_ = true;
  while (running_) {
    try {
      // Pull the next planner task from the queue
      PlannerTask task = planner_queue_.pull();

      LOG_DEBUG("Executor received task for copy_op_id: {}", task.copy_op_id);

      // Compile the CopySpec into a Plan using NCCLPlanner
      Plan plan = planner_.Compile(task.copy_spec, metastore_);

      LOG_DEBUG("Compiled plan:\n{}", plan);

      // Fragment the plan by NodeId
      auto fragments = plan.Fragments();

      // Send ExecuteRequest to each node agent's async (DEALER) socket
      for (auto& [node_id, node_plan] : fragments) {
        Identity node_identity = boost::uuids::to_string(node_id) + "_dealer";

        ExecuteRequest execute_request(task.copy_op_id, std::move(node_plan));

        outbox_queue_.push(OutboxMessage{node_identity, execute_request});
      }

      // Set expected responses (atomic write with release ordering for
      // visibility to Handler thread)
      task.state->expected_responses.store(fragments.size(),
                                           std::memory_order_release);
      LOG_DEBUG("Executor: Set expected_responses={} for copy_op_id={}",
                fragments.size(), task.copy_op_id);

    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
