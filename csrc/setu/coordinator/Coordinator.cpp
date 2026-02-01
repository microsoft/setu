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
using setu::commons::datatypes::TensorShardRef;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::NodeAgentRequest;
using setu::commons::messages::RegisterTensorShardResponse;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::utils::Comm;
using setu::commons::utils::ZmqHelper;
//==============================================================================
constexpr std::int32_t kPollTimeoutMs = 100;
constexpr std::chrono::milliseconds kExecutorLoopSleepMs(10);
//==============================================================================
// Coordinator Implementation
//==============================================================================
Coordinator::Coordinator(std::size_t port)
    : port_(port), zmq_context_(std::make_shared<zmq::context_t>()) {
  gateway_ = std::make_unique<Gateway>(zmq_context_, port_, inbox_queue_,
                                       outbox_queue_);
  handler_ = std::make_unique<Handler>(inbox_queue_, outbox_queue_, metastore_,
                                       planner_queue_);
  executor_ = std::make_unique<Executor>(outbox_queue_);
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
  outbox_queue_.close();

  gateway_->Stop();
  handler_->Stop();
  executor_->Stop();
}

std::optional<TensorShardRef> Coordinator::RegisterTensorShard(
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
                              Queue<CopySpec>& planner_queue)
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
      std::visit(
          [&](const auto& msg) {
            using T = std::decay_t<decltype(msg)>;
            if constexpr (std::is_same_v<T, RegisterTensorShardRequest>) {
              HandleRegisterTensorShardRequest(inbox_msg.node_agent_identity,
                                               msg);
            } else if constexpr (std::is_same_v<T, SubmitCopyRequest>) {
              HandleSubmitCopyRequest(inbox_msg.node_agent_identity, msg);
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
  TensorShardRef shard_ref =
      metastore_.RegisterTensorShard(request.tensor_shard_spec, owner_node_id);

  // Send response
  RegisterTensorShardResponse response(request.request_id, ErrorCode::kSuccess,
                                       shard_ref);
  outbox_queue_.push(OutboxMessage{node_agent_identity, response});

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

    // Send AllocateTensorRequest to all NodeAgents that own shards
    AllocateTensorRequest allocate_request(request.tensor_shard_spec.name);
    for (const NodeId& owner_id : metadata->GetOwnerNodeIds()) {
      Identity owner_identity = to_string(owner_id);
      outbox_queue_.push(OutboxMessage{owner_identity, allocate_request});
    }
  }
}

void Coordinator::Handler::HandleSubmitCopyRequest(
    const Identity& node_agent_identity, const SubmitCopyRequest& request) {
  LOG_INFO("Coordinator received SubmitCopyRequest from {} to {}",
           request.copy_spec.src_name, request.copy_spec.dst_name);

  CopyKey copy_key{request.copy_spec.src_name, request.copy_spec.dst_name};

  // Check if this is the first request for this (src, dst) pair
  auto pending_it = pending_copy_specs_.find(copy_key);
  if (pending_it == pending_copy_specs_.end()) {
    // First request - store the CopySpec for validation
    pending_copy_specs_.emplace(copy_key, request.copy_spec);
    copies_received_[copy_key] = 1;
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

    copies_received_[copy_key]++;
  }

  // Track this node agent for later response
  pending_node_agents_[copy_key].push_back(
      PendingNodeAgent{node_agent_identity, request.request_id});

  // Get the expected number of clients (number of shards for source tensor)
  std::size_t expected_clients =
      metastore_.GetNumShardsForTensor(request.copy_spec.src_name);

  LOG_DEBUG("SubmitCopy {} -> {}: received {}/{} requests",
            request.copy_spec.src_name, request.copy_spec.dst_name,
            copies_received_[copy_key], expected_clients);

  // Check if all clients have sent the request
  if (copies_received_[copy_key] == expected_clients) {
    // Generate CopyOperationId
    CopyOperationId copy_op_id = GenerateUUID();

    LOG_INFO(
        "All clients submitted copy request {} -> {}, "
        "copy_op_id={}, adding to planner queue",
        request.copy_spec.src_name, request.copy_spec.dst_name, copy_op_id);

    // Store the mapping
    copy_operations_.emplace(copy_op_id, request.copy_spec);

    // Add to planner queue
    planner_queue_.push(request.copy_spec);

    // Send responses to all waiting clients with copy_op_id
    for (const auto& client : pending_node_agents_[copy_key]) {
      SubmitCopyResponse response(client.request_id, copy_op_id,
                                  ErrorCode::kSuccess);
      outbox_queue_.push(OutboxMessage{client.identity, response});
    }

    // Clean up maps
    copies_received_.erase(copy_key);
    pending_copy_specs_.erase(copy_key);
    pending_node_agents_.erase(copy_key);
  }
}

//==============================================================================
// Executor Implementation
//==============================================================================
Coordinator::Executor::Executor(Queue<OutboxMessage>& outbox_queue)
    : outbox_queue_(outbox_queue) {}

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
    // TODO: Implement executor loop to dispatch plans to NodeAgents
    // Will pull from an executor_queue_ and push to outbox_queue_
    // For now, just sleep and check running_ flag
    std::this_thread::sleep_for(kExecutorLoopSleepMs);
    if (outbox_queue_.closed()) {
      LOG_DEBUG("Executor: outbox_queue_ closed, exiting");
      return;
    }
  }
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
