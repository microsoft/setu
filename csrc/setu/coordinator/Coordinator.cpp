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
  executor_ = std::make_unique<Executor>(planner_queue_, outbox_queue_, metastore_);
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
  LOG_DEBUG("Gateway: Initializing ZMQ sockets");

  node_agent_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, port_);

  LOG_DEBUG("Gateway: Initialized ZMQ sockets successfully");
}

void Coordinator::Gateway::CloseSockets() {
  LOG_DEBUG("Gateway: Closing ZMQ sockets");

  if (node_agent_socket_) {
    node_agent_socket_->close();
  }

  LOG_DEBUG("Gateway: Closed ZMQ sockets successfully");
}

void Coordinator::Gateway::Start() {
  if (running_.load()) {
    return;
  }
  LOG_DEBUG("Starting gateway loop");
  thread_ = std::thread(SETU_LAUNCH_THREAD([this]() { this->Loop(); },
                                           "CoordinatorGatewayThread"));
}

void Coordinator::Gateway::Stop() {
  LOG_DEBUG("Stopping gateway loop");
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
  LOG_DEBUG("Gateway loop stopped");
}

void Coordinator::Gateway::Loop() {
  LOG_DEBUG("Entering gateway loop");

  running_ = true;
  while (running_) {
    // Poll for incoming messages from NodeAgents
    auto ready = Comm::PollForRead({node_agent_socket_}, kPollTimeoutMs);

    for (const auto& socket : ready) {
      if (socket == node_agent_socket_) {
        auto [node_agent_identity, request] =
            Comm::RecvWithIdentity<NodeAgentRequest, false>(socket);
        LOG_DEBUG("Gateway: Received message from {} (variant index={})",
                  node_agent_identity, request.index());
        auto status =
            inbox_queue_.try_push(InboxMessage{node_agent_identity, request});
        if (status == boost::queue_op_status::closed) {
          LOG_DEBUG("Gateway: inbox_queue_ closed, exiting");
          return;
        }
      }
    }

    // Send any outgoing messages (drain all available without blocking)
    try {
      while (!outbox_queue_.empty()) {
        OutboxMessage outbox_msg = outbox_queue_.pull();
        Comm::SendWithIdentity<CoordinatorMessage, false>(
            node_agent_socket_, outbox_msg.node_agent_identity,
            outbox_msg.message);
      }
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      LOG_DEBUG("Gateway: outbox_queue_ closed, exiting");
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
  LOG_DEBUG("Starting handler loop");
  thread_ = std::thread(SETU_LAUNCH_THREAD([this]() { this->Loop(); },
                                           "CoordinatorHandlerThread"));
}

void Coordinator::Handler::Stop() {
  LOG_DEBUG("Stopping handler loop");
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
  LOG_DEBUG("Handler loop stopped");
}

void Coordinator::Handler::Loop() {
  LOG_DEBUG("Entering handler loop");

  running_ = true;
  while (running_) {
    try {
      InboxMessage inbox_msg = inbox_queue_.pull();
      LOG_DEBUG("Handler: Processing message from {} (variant index={})",
                inbox_msg.node_agent_identity, inbox_msg.request.index());
      std::visit(
          [&](const auto& msg) {
            using T = std::decay_t<decltype(msg)>;
            if constexpr (std::is_same_v<T, RegisterTensorShardRequest>) {
              LOG_DEBUG("Handler: Dispatching RegisterTensorShardRequest");
              HandleRegisterTensorShardRequest(inbox_msg.node_agent_identity,
                                               msg);
            } else if constexpr (std::is_same_v<T, SubmitCopyRequest>) {
              LOG_DEBUG("Handler: Dispatching SubmitCopyRequest");
              HandleSubmitCopyRequest(inbox_msg.node_agent_identity, msg);
            } else if constexpr (std::is_same_v<T, SubmitPullRequest>) {
              LOG_DEBUG("Handler: Dispatching SubmitPullRequest");
              HandleSubmitPullRequest(inbox_msg.node_agent_identity, msg);
            } else if constexpr (std::is_same_v<T, ExecuteResponse>) {
              LOG_DEBUG("Handler: Dispatching ExecuteResponse");
              HandleExecuteResponse(inbox_msg.node_agent_identity, msg);
            } else {
              LOG_WARNING("Handler: Unknown message type (index={})",
                          inbox_msg.request.index());
            }
          },
          inbox_msg.request);
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      LOG_DEBUG("Handler: inbox_queue_ closed, exiting");
      return;
    }
  }
}

void Coordinator::Handler::HandleRegisterTensorShardRequest(
    const Identity& node_agent_identity,
    const RegisterTensorShardRequest& request) {
  LOG_INFO("Coordinator received RegisterTensorShardRequest for tensor: {}",
           request.tensor_shard_spec.name);

  // Parse NodeId from the identity (NodeAgent uses to_string(node_id) as
  // identity)
  NodeId owner_node_id = StringToUUID(node_agent_identity);

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

  AllocateTensorRequest allocate_request({shard_metadata_ptr->id});
  outbox_queue_.push(OutboxMessage{to_string(owner_node_id), allocate_request});

  // // Check if all shards for this tensor are registered
  // if (metastore_.AllShardsRegistered(request.tensor_shard_spec.name)) {
  //   LOG_INFO(
  //       "All shards registered for tensor: {}, sending AllocateTensorRequest "
  //       "to all owners",
  //       request.tensor_shard_spec.name);

  //   // Get tensor metadata to find all owner NodeIds
  //   auto metadata =
  //       metastore_.GetTensorMetadata(request.tensor_shard_spec.name);
  //   ASSERT_VALID_POINTER_ARGUMENT(metadata);

  //   // Group shard IDs by owner node
  //   std::unordered_map<NodeId, std::vector<ShardId>> owner_to_shard_ids;
  //   for (const auto& [shard_id, shard_metadata] : metadata->shards) {
  //     owner_to_shard_ids[shard_metadata->owner].push_back(shard_id);
  //   }

  //   // Send AllocateTensorRequest to each NodeAgent with its shard IDs
  //   for (const auto& [owner_id, shard_ids] : owner_to_shard_ids) {
  //     Identity owner_identity = to_string(owner_id);
  //     AllocateTensorRequest allocate_request(shard_ids);
  //     outbox_queue_.push(OutboxMessage{owner_identity, allocate_request});
  //   }
  // }
}

void Coordinator::Handler::HandleSubmitCopyRequest(
    const Identity& node_agent_identity, const SubmitCopyRequest& request) {
  LOG_INFO("Coordinator received SubmitCopyRequest from {} to {} for shard {}",
           request.copy_spec.src_name, request.copy_spec.dst_name,
           request.shard_id);

  CopyKey copy_key{request.copy_spec.src_name, request.copy_spec.dst_name};

  // Track which shards have submitted
  auto& shards = shards_received_[copy_key];

  // Check for duplicate shard submission
  ASSERT_VALID_RUNTIME(
      !shards.contains(request.shard_id),
      "SubmitCopy {} -> {}: duplicate submission from shard {}",
      request.copy_spec.src_name, request.copy_spec.dst_name, request.shard_id);

  // Check if this is the first request for this (src, dst) pair
  auto pending_it = pending_copy_specs_.find(copy_key);
  if (pending_it == pending_copy_specs_.end()) {
    // First request - store the CopySpec for validation
    pending_copy_specs_.emplace(copy_key, request.copy_spec);
  } else {
    // Subsequent request - verify TensorSelections match
    const CopySpec& first_spec = pending_it->second;

    /// TODO: need to handle errors differently
    ASSERT_VALID_RUNTIME(
        *request.copy_spec.src_selection == *first_spec.src_selection,
        "SubmitCopy {} -> {}: source selection mismatch",
        request.copy_spec.src_name, request.copy_spec.dst_name);

    ASSERT_VALID_RUNTIME(
        *request.copy_spec.dst_selection == *first_spec.dst_selection,
        "SubmitCopy {} -> {}: destination selection mismatch",
        request.copy_spec.src_name, request.copy_spec.dst_name);
  }

  shards.insert(request.shard_id);

  // Track this node agent for later response
  pending_node_agents_[copy_key].push_back(
      PendingNodeAgent{node_agent_identity, request.request_id});

  // Expected = all src shards + all dst shards
  std::size_t expected_shards =
      metastore_.GetNumShardsForTensor(request.copy_spec.src_name) +
      metastore_.GetNumShardsForTensor(request.copy_spec.dst_name);

  LOG_DEBUG("SubmitCopy {} -> {}: received {}/{} shards",
            request.copy_spec.src_name, request.copy_spec.dst_name,
            shards.size(), expected_shards);

  // Check if all shards have submitted
  if (shards.size() == expected_shards) {
    // Generate CopyOperationId
    CopyOperationId copy_op_id = GenerateUUID();

    LOG_INFO(
        "All shards submitted copy request {} -> {}, "
        "copy_op_id={}, adding to planner queue",
        request.copy_spec.src_name, request.copy_spec.dst_name, copy_op_id);

    // Collect submitter identities before cleaning up pending_node_agents_
    std::vector<Identity> submitters;
    for (const auto& client : pending_node_agents_[copy_key]) {
      submitters.push_back(client.identity);
    }

    // Create shared state with submitter identities
    auto state = std::make_shared<CopyOperationState>(request.copy_spec,
                                                      std::move(submitters));

    // Store the shared state (will be accessed by HandleExecuteResponse)
    copy_operations_.emplace(copy_op_id, state);

    // Add to planner queue with copy_op_id and shared state
    planner_queue_.push(PlannerTask{copy_op_id, request.copy_spec, state});

    // Send responses to all waiting clients with copy_op_id
    for (const auto& client : pending_node_agents_[copy_key]) {
      SubmitCopyResponse response(client.request_id, copy_op_id,
                                  ErrorCode::kSuccess);
      outbox_queue_.push(OutboxMessage{client.identity, response});
    }

    // Clean up maps
    shards_received_.erase(copy_key);
    pending_copy_specs_.erase(copy_key);
    pending_node_agents_.erase(copy_key);
  }
}

void Coordinator::Handler::HandleSubmitPullRequest(
    const Identity& node_agent_identity, const SubmitPullRequest& request) {
  LOG_INFO("Coordinator received SubmitPullRequest from {} to {} for shard {}",
           request.copy_spec.src_name, request.copy_spec.dst_name,
           request.shard_id);

  CopyKey copy_key{request.copy_spec.src_name, request.copy_spec.dst_name};

  // Track which shards have submitted
  auto& shards = shards_received_[copy_key];

  // Check for duplicate shard submission
  ASSERT_VALID_RUNTIME(
      !shards.contains(request.shard_id),
      "SubmitPull {} -> {}: duplicate submission from shard {}",
      request.copy_spec.src_name, request.copy_spec.dst_name, request.shard_id);

  // Check if this is the first request for this (src, dst) pair
  auto pending_it = pending_copy_specs_.find(copy_key);
  if (pending_it == pending_copy_specs_.end()) {
    // First request - store the CopySpec for validation
    pending_copy_specs_.emplace(copy_key, request.copy_spec);
  } else {
    // Subsequent request - verify TensorSelections match
    const CopySpec& first_spec = pending_it->second;

    /// TODO: need to handle errors differently
    ASSERT_VALID_RUNTIME(
        *request.copy_spec.src_selection == *first_spec.src_selection,
        "SubmitPull {} -> {}: source selection mismatch",
        request.copy_spec.src_name, request.copy_spec.dst_name);

    ASSERT_VALID_RUNTIME(
        *request.copy_spec.dst_selection == *first_spec.dst_selection,
        "SubmitPull {} -> {}: destination selection mismatch",
        request.copy_spec.src_name, request.copy_spec.dst_name);
  }

  shards.insert(request.shard_id);

  // Track this node agent for later response
  pending_node_agents_[copy_key].push_back(
      PendingNodeAgent{node_agent_identity, request.request_id});

  // For Pull: expected shards = number of DESTINATION shards only (one-sided)
  std::size_t expected_shards =
      metastore_.GetNumShardsForTensor(request.copy_spec.dst_name);

  LOG_DEBUG("SubmitPull {} -> {}: received {}/{} shards",
            request.copy_spec.src_name, request.copy_spec.dst_name,
            shards.size(), expected_shards);

  // Check if all shards have submitted
  if (shards.size() == expected_shards) {
    // Generate CopyOperationId
    CopyOperationId copy_op_id = GenerateUUID();

    LOG_INFO(
        "All shards submitted pull request {} -> {}, "
        "copy_op_id={}, adding to planner queue",
        request.copy_spec.src_name, request.copy_spec.dst_name, copy_op_id);

    // Collect submitter identities before cleaning up pending_node_agents_
    std::vector<Identity> submitters;
    for (const auto& client : pending_node_agents_[copy_key]) {
      submitters.push_back(client.identity);
    }

    // Create shared state with submitter identities
    auto state = std::make_shared<CopyOperationState>(request.copy_spec,
                                                      std::move(submitters));

    // Store the shared state (will be accessed by HandleExecuteResponse)
    copy_operations_.emplace(copy_op_id, state);

    // Add to planner queue with copy_op_id and shared state
    planner_queue_.push(PlannerTask{copy_op_id, request.copy_spec, state});

    // Send responses to all waiting clients with copy_op_id
    for (const auto& client : pending_node_agents_[copy_key]) {
      SubmitCopyResponse response(client.request_id, copy_op_id,
                                  ErrorCode::kSuccess);
      outbox_queue_.push(OutboxMessage{client.identity, response});
    }

    // Clean up maps
    shards_received_.erase(copy_key);
    pending_copy_specs_.erase(copy_key);
    pending_node_agents_.erase(copy_key);
  }
}

void Coordinator::Handler::HandleExecuteResponse(
    const Identity& node_identity, const ExecuteResponse& response) {
  LOG_DEBUG("Handling ExecuteResponse from {} for copy_op_id: {}",
            node_identity, response.copy_op_id);

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
    LOG_INFO("All {} participants completed for copy_op_id {}, notifying {} "
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
      metastore_(metastore) {}

void Coordinator::Executor::Start() {
  if (running_.load()) {
    return;
  }
  LOG_DEBUG("Starting executor loop");
  thread_ = std::thread(SETU_LAUNCH_THREAD([this]() { this->Loop(); },
                                           "CoordinatorExecutorThread"));
}

void Coordinator::Executor::Stop() {
  LOG_DEBUG("Stopping executor loop");
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
  LOG_DEBUG("Executor loop stopped");
}

void Coordinator::Executor::Loop() {
  LOG_DEBUG("Entering executor loop");

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

      LOG_DEBUG("Fragmented plan into {} node-specific plans", fragments.size());

      // Send ExecuteRequest to each node agent
      for (auto& [node_id, node_plan] : fragments) {
        Identity node_identity = boost::uuids::to_string(node_id);

        LOG_DEBUG("Sending ExecuteRequest to node {} for fragment {} with copy_op_id: {}",
                  node_identity, node_plan, task.copy_op_id);

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
      LOG_DEBUG("Executor: planner_queue_ closed, exiting");
      return;
    }
  }
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
