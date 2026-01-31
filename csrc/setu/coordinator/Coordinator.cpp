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
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::datatypes::TensorDim;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::NodeAgentRequest;
using setu::commons::messages::RegisterTensorShardResponse;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::WaitForCopyResponse;
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
  handler_ = std::make_unique<Handler>(inbox_queue_, outbox_queue_);
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
                              Queue<OutboxMessage>& outbox_queue)
    : inbox_queue_(inbox_queue), outbox_queue_(outbox_queue) {}

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
          [&](const auto& req) {
            HandleNodeAgentRequest(inbox_msg.node_agent_identity, req);
          },
          inbox_msg.request);
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      LOG_DEBUG("Handler: inbox_queue_ closed, exiting");
      return;
    }
  }
}

void Coordinator::Handler::HandleNodeAgentRequest(
    const Identity& node_agent_identity,
    const RegisterTensorShardRequest& request) {
  LOG_INFO("Coordinator received RegisterTensorShardRequest for tensor: {}",
           request.tensor_shard_spec.name);

  // Generate a shard ID
  ShardId shard_id = GenerateUUID();

  // Build TensorDimMap from the spec's dims (using owned size for shard ref)
  TensorDimMap dim_map;
  for (const auto& dim_spec : request.tensor_shard_spec.dims) {
    dim_map.emplace(dim_spec.name,
                    TensorDim(dim_spec.name, dim_spec.GetOwnedSize()));
  }

  // Create TensorShardRef
  TensorShardRef shard_ref(request.tensor_shard_spec.name, shard_id, dim_map);

  // Send response to client via outbox
  RegisterTensorShardResponse response(request.request_id, ErrorCode::kSuccess,
                                       shard_ref);
  outbox_queue_.push(OutboxMessage{node_agent_identity, response});

  // Send AllocateTensorRequest to NodeAgent to allocate the tensor
  AllocateTensorRequest allocate_request(request.tensor_shard_spec.name);
  outbox_queue_.push(OutboxMessage{node_agent_identity, allocate_request});

  LOG_INFO("Queued AllocateTensorRequest for tensor: {}",
           request.tensor_shard_spec.name);
}

void Coordinator::Handler::HandleNodeAgentRequest(
    const Identity& node_agent_identity, const SubmitCopyRequest& request) {
  LOG_INFO("Coordinator received SubmitCopyRequest from {} to {}",
           request.copy_spec.src_name, request.copy_spec.dst_name);

  // TODO: Actually submit the copy operation
  // For now, just log and respond with success
  LOG_INFO("Submitted copy operation: {} -> {} (stub implementation)",
           request.copy_spec.src_name, request.copy_spec.dst_name);

  SubmitCopyResponse response(RequestId(), ErrorCode::kSuccess);
  outbox_queue_.push(OutboxMessage{node_agent_identity, response});
}

void Coordinator::Handler::HandleNodeAgentRequest(
    const Identity& node_agent_identity, const WaitForCopyRequest& request) {
  LOG_INFO("Coordinator received WaitForCopyRequest for copy operation ID: {}",
           request.copy_operation_id);

  // TODO: Actually wait for the copy operation
  // For now, just log and respond with success
  LOG_INFO("WaitForCopy: {} (stub implementation)", request.copy_operation_id);

  WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);
  outbox_queue_.push(OutboxMessage{node_agent_identity, response});
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
