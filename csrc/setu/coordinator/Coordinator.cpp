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
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::CopyOperationFinishedRequest;
using setu::commons::messages::DeregisterShardsRequest;
using setu::commons::messages::DeregisterShardsResponse;
using setu::commons::messages::ExecuteRequest;
using setu::commons::messages::ExecuteResponse;
using setu::commons::messages::GetTensorSpecRequest;
using setu::commons::messages::GetTensorSpecResponse;
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
Coordinator::Coordinator(std::size_t port, PlannerPtr planner)
    : port_(port),
      zmq_context_(std::make_shared<zmq::context_t>()),
      planner_(planner) {
  gateway_ = std::make_unique<Gateway>(zmq_context_, port_, inbox_queue_,
                                       outbox_queue_);

  auto outbox_notify = [this]() { gateway_->NotifyOutbox(); };

  handler_ = std::make_unique<Handler>(inbox_queue_, outbox_queue_, metastore_,
                                       planner_queue_, outbox_notify);
  executor_ =
      std::make_unique<Executor>(planner_queue_, outbox_queue_, metastore_,
                                 *planner_, hint_store_, outbox_notify);
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

void Coordinator::AddHint(setu::planner::hints::CompilerHint hint) {
  hint_store_.AddHint(std::move(hint));
}

void Coordinator::ClearHints() { hint_store_.Clear(); }

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

  // Create inproc PAIR sockets for self-pipe wakeup pattern
  wakeup_recv_ =
      std::make_shared<zmq::socket_t>(*zmq_context_, zmq::socket_type::pair);
  wakeup_send_ =
      std::make_shared<zmq::socket_t>(*zmq_context_, zmq::socket_type::pair);
  wakeup_recv_->bind("inproc://gateway-wakeup");
  wakeup_send_->connect("inproc://gateway-wakeup");
}

void Coordinator::Gateway::NotifyOutbox() {
  // Send a single byte to wake the Gateway from zmq::poll().
  // Uses dontwait to avoid blocking the caller if the pipe is full.
  zmq::message_t signal(1);
  static_cast<char*>(signal.data())[0] = 'W';
  [[maybe_unused]] auto result =
      wakeup_send_->send(std::move(signal), zmq::send_flags::dontwait);
}

void Coordinator::Gateway::CloseSockets() {
  if (wakeup_send_) wakeup_send_->close();
  if (wakeup_recv_) wakeup_recv_->close();
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
    // Poll for incoming messages from NodeAgents OR wakeup signal
    auto ready =
        Comm::PollForRead({node_agent_socket_, wakeup_recv_}, kPollTimeoutMs);

    for (const auto& socket : ready) {
      if (socket == node_agent_socket_) {
        auto [node_agent_identity, request] =
            Comm::RecvWithIdentity<NodeAgentRequest>(socket);
        auto status =
            inbox_queue_.try_push(InboxMessage{node_agent_identity, request});
        if (status == boost::queue_op_status::closed) {
          return;
        }
      } else if (socket == wakeup_recv_) {
        // Drain all wakeup signals (there may be multiple)
        zmq::message_t drain;
        while (
            wakeup_recv_->recv(drain, zmq::recv_flags::dontwait).has_value()) {
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
                              Queue<PlannerTask>& planner_queue,
                              OutboxNotifyFn outbox_notify)
    : inbox_queue_(inbox_queue),
      outbox_queue_(outbox_queue),
      metastore_(metastore),
      planner_queue_(planner_queue),
      outbox_notify_(std::move(outbox_notify)) {}

void Coordinator::Handler::PushOutbox(OutboxMessage msg) {
  outbox_queue_.push(std::move(msg));
  outbox_notify_();
}

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
            } else if constexpr (std::is_same_v<T, GetTensorSpecRequest>) {
              HandleGetTensorSpecRequest(inbox_msg.node_agent_identity, msg);
            } else if constexpr (std::is_same_v<T, DeregisterShardsRequest>) {
              HandleDeregisterShardsRequest(inbox_msg.node_agent_identity, msg);
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

  // Reject registration if the tensor is being deregistered
  if (metastore_.IsTensorDeregistered(request.tensor_shard_spec.name)) {
    LOG_WARNING(
        "Rejecting RegisterTensorShardRequest: tensor '{}' has deregistered "
        "shards",
        request.tensor_shard_spec.name);
    RegisterTensorShardCoordinatorResponse response(
        request.request_id, ErrorCode::kTensorDeregistered);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

  // Register the tensor shard in the metastore with owner information
  auto shard_metadata_ptr =
      metastore_.RegisterTensorShard(request.tensor_shard_spec, owner_node_id);

  // Send response with TensorShardMetadata
  if (shard_metadata_ptr) {
    RegisterTensorShardCoordinatorResponse response(
        request.request_id, ErrorCode::kSuccess, *shard_metadata_ptr);
    PushOutbox(OutboxMessage{node_agent_identity, response});
  } else {
    LOG_ERROR("Failed to register tensor shard: {}", request.tensor_shard_spec);
    RegisterTensorShardCoordinatorResponse response(
        request.request_id, ErrorCode::kInvalidArguments);
    PushOutbox(OutboxMessage{node_agent_identity, response});
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
      PushOutbox(OutboxMessage{owner_identity, allocate_request});
    }
  }
}

void Coordinator::Handler::HandleSubmitCopyRequest(
    const Identity& node_agent_identity, const SubmitCopyRequest& request) {
  LOG_INFO("Coordinator received SubmitCopyRequest from {} to {} for shard {}",
           request.copy_spec.src_name, request.copy_spec.dst_name,
           request.shard_id);

  if (metastore_.IsTensorDeregistered(request.copy_spec.src_name) ||
      metastore_.IsTensorDeregistered(request.copy_spec.dst_name)) {
    LOG_WARNING(
        "Rejecting SubmitCopyRequest: tensor '{}' or '{}' has deregistered "
        "shards",
        request.copy_spec.src_name, request.copy_spec.dst_name);
    SubmitCopyResponse response(request.request_id, CopyOperationId{},
                                ErrorCode::kTensorDeregistered);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

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

  if (metastore_.IsTensorDeregistered(request.copy_spec.src_name) ||
      metastore_.IsTensorDeregistered(request.copy_spec.dst_name)) {
    LOG_WARNING(
        "Rejecting SubmitPullRequest: tensor '{}' or '{}' has deregistered "
        "shards",
        request.copy_spec.src_name, request.copy_spec.dst_name);
    SubmitCopyResponse response(request.request_id, CopyOperationId{},
                                ErrorCode::kTensorDeregistered);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

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
      "All {} shards submitted for {} -> {}, "
      "copy_op_id={}, adding to planner queue",
      expected_shards, copy_spec.src_name, copy_spec.dst_name, copy_op_id);

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
    PushOutbox(OutboxMessage{participant.identity, response});
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
      PushOutbox(OutboxMessage{submitter_identity, finish_req});
    }

    copy_operations_.erase(it);

    // Check if any deferred deregistrations are now unblocked
    auto unblocked = deregistration_tracker_.Resolve(response.copy_op_id);
    for (auto& dereg : unblocked) {
      LOG_INFO(
          "All blocking copies completed for deregistration from {} — "
          "proceeding with deregistration",
          dereg.node_agent_identity);

      metastore_.DeregisterShards(dereg.shards_by_tensor);

      DeregisterShardsResponse dereg_response(dereg.request_id,
                                              ErrorCode::kSuccess);
      outbox_queue_.push(
          OutboxMessage{dereg.node_agent_identity, dereg_response});
    }
  }
}

void Coordinator::Handler::HandleGetTensorSpecRequest(
    const Identity& node_agent_identity, const GetTensorSpecRequest& request) {
  LOG_DEBUG("Coordinator received GetTensorSpecRequest for tensor: {}",
            request.tensor_name);

  if (metastore_.IsTensorDeregistered(request.tensor_name)) {
    LOG_WARNING(
        "Rejecting GetTensorSpecRequest: tensor '{}' has deregistered shards",
        request.tensor_name);
    GetTensorSpecResponse response(request.request_id,
                                   ErrorCode::kTensorDeregistered);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

  const auto* tensor_spec = metastore_.GetTensorSpec(request.tensor_name);
  ASSERT_VALID_RUNTIME(
      tensor_spec != nullptr,
      "TensorSpec must exist for tensor '{}' — at least one shard should have "
      "been registered before GetTensorSpecRequest is sent",
      request.tensor_name);

  GetTensorSpecResponse response(request.request_id, ErrorCode::kSuccess,
                                 *tensor_spec);
  PushOutbox(OutboxMessage{node_agent_identity, response});
}

/// Deregistration is two-phase (see MetaStore class docstring):
///
///   1. Mark. Immediately mark all affected tensors as deregistered so new
///      registrations, copies, and pulls are rejected. Cancel any partial
///      shard aggregation groups that reference these tensors.
///
///   2. Remove. Remove shard metadata from the MetaStore and send a
///      success response to the requesting NodeAgent. If there are
///      in-flight copy operations touching these tensors, defer this step
///      using deregistration_tracker_ (a PendingOperations instance) which
///      releases the deregistration request once all blocking copy
///      operations have finished. On receiving the response, the NodeAgent
///      cleans up its own local state (shard mappings, caches, blocker
///      registrations) and forwards the response to the client.
void Coordinator::Handler::HandleDeregisterShardsRequest(
    const Identity& node_agent_identity,
    const DeregisterShardsRequest& request) {
  LOG_INFO("Coordinator received DeregisterShardsRequest from {}",
           node_agent_identity);

  // Collect tensor names being deregistered
  std::set<TensorName> tensor_names;
  for (const auto& [name, _] : request.shards_by_tensor) {
    tensor_names.insert(name);
  }

  // Mark tensors as deregistered immediately to prevent new copy/pull
  // submissions and registrations from being accepted while deregistration
  // is in progress (even if the actual shard removal is deferred).
  for (const auto& name : tensor_names) {
    metastore_.MarkTensorDeregistered(name);
  }

  // Cancel partial entries in the shard aggregator for these tensors.
  // This cleans up groups that will never complete because the shards are
  // going away.
  auto cancelled_participants =
      shard_aggregator_.CancelIf([&tensor_names](const CopyKey& key) {
        return tensor_names.contains(key.src_name) ||
               tensor_names.contains(key.dst_name);
      });

  // Send error responses to cancelled participants
  for (const auto& participant : cancelled_participants) {
    LOG_INFO(
        "Cancelling pending copy submission for participant {} due to tensor "
        "deregistration",
        participant.identity);
    SubmitCopyResponse error_response(participant.request_id, CopyOperationId{},
                                      ErrorCode::kTensorDeregistered);
    outbox_queue_.push(OutboxMessage{participant.identity, error_response});
  }

  // Find all in-flight copy operations that involve any of the tensors
  // being deregistered by scanning active copy operations
  std::set<CopyOperationId> blocking_ops;
  for (const auto& [copy_op_id, state] : copy_operations_) {
    if (tensor_names.contains(state->spec.src_name) ||
        tensor_names.contains(state->spec.dst_name)) {
      blocking_ops.insert(copy_op_id);
    }
  }

  PendingDeregistration dereg_data{node_agent_identity, request.request_id,
                                   request.shards_by_tensor};

  if (blocking_ops.empty()) {
    // No in-flight copies — deregister immediately
    metastore_.DeregisterShards(dereg_data.shards_by_tensor);
    DeregisterShardsResponse response(request.request_id, ErrorCode::kSuccess);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

  auto immediate = deregistration_tracker_.AddWaiter(
      request.request_id, std::move(blocking_ops), std::move(dereg_data));

  if (immediate.has_value()) {
    // All blocking copies already resolved — deregister immediately
    metastore_.DeregisterShards(immediate->shards_by_tensor);
    DeregisterShardsResponse response(request.request_id, ErrorCode::kSuccess);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
  } else {
    LOG_INFO(
        "Deferring deregistration for {} tensors from {} — blocked by "
        "in-flight copy operations",
        tensor_names.size(), node_agent_identity);
  }
}

//==============================================================================
// Executor Implementation
//==============================================================================
Coordinator::Executor::Executor(Queue<PlannerTask>& planner_queue,
                                Queue<OutboxMessage>& outbox_queue,
                                MetaStore& metastore, Planner& planner,
                                HintStore& hint_store,
                                OutboxNotifyFn outbox_notify)
    : planner_queue_(planner_queue),
      outbox_queue_(outbox_queue),
      metastore_(metastore),
      planner_(planner),
      hint_store_(hint_store),
      outbox_notify_(std::move(outbox_notify)) {}

void Coordinator::Executor::PushOutbox(OutboxMessage msg) {
  outbox_queue_.push(std::move(msg));
  outbox_notify_();
}

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
      PlannerTask task = planner_queue_.pull();
      auto t_after_dequeue = std::chrono::steady_clock::now();

      LOG_DEBUG("Executor received task for copy_op_id: {}", task.copy_op_id);

      auto hints = hint_store_.Snapshot();
      auto t_compile_start = std::chrono::steady_clock::now();
      Plan plan = planner_.Compile(task.copy_spec, metastore_, hints);
      auto t_compile_end = std::chrono::steady_clock::now();

      LOG_DEBUG("Compiled plan:\n{}", plan);

      // Fragment the plan to into per-node fragments
      auto fragments = plan.Fragments();

      // Send ExecuteRequest to each node agent
      for (auto& [node_id, node_plan] : fragments) {
        Identity node_identity = boost::uuids::to_string(node_id) + "_dealer";

        ExecuteRequest execute_request(task.copy_op_id, std::move(node_plan));

        PushOutbox(OutboxMessage{node_identity, execute_request});
      }

      // Set expected responses
      // memory order release so Handler thread can pick it up (using memory
      // order aqcuire)
      task.state->expected_responses.store(fragments.size(),
                                           std::memory_order_release);

      auto t_end = std::chrono::steady_clock::now();
      auto to_us = [](auto d) {
        return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
      };
      LOG_INFO(
          "Executor: copy_op_id={}, compile={}us, fragment+dispatch={}us, "
          "total={}us",
          task.copy_op_id, to_us(t_compile_end - t_compile_start),
          to_us(t_end - t_compile_end), to_us(t_end - t_after_dequeue));

    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
