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
#include "commons/utils/SetuCommHelper.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::datatypes::TensorDim;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorShardIdentifier;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::NodeAgentRequest;
using setu::commons::messages::RegisterTensorShardResponse;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::WaitForCopyResponse;
using setu::commons::utils::SetuCommHelper;
using setu::commons::utils::ZmqHelper;
//==============================================================================
constexpr std::chrono::milliseconds kHandleLoopSleepMs(10);
//==============================================================================
Coordinator::Coordinator(std::size_t router_executor_port,
                         std::size_t router_handler_port)
    : router_executor_port_(router_executor_port),
      router_handler_port_(router_handler_port) {
  InitZmqSockets();
}

Coordinator::~Coordinator() {
  Stop();
  CloseZmqSockets();
}

void Coordinator::Start() {
  LOG_DEBUG("Starting Coordinator");
  StartHandlerLoop();
  StartExecutorLoop();
}

void Coordinator::Stop() {
  LOG_DEBUG("Stopping Coordinator");
  StopHandlerLoop();
  StopExecutorLoop();
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

void Coordinator::InitZmqSockets() {
  LOG_DEBUG("Initializing ZMQ sockets");

  zmq_context_ = std::make_shared<zmq::context_t>();

  node_agent_router_executor_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, router_executor_port_);
  node_agent_router_handler_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, router_handler_port_);

  LOG_DEBUG("Initialized ZMQ sockets successfully");
}

void Coordinator::CloseZmqSockets() {
  LOG_DEBUG("Closing ZMQ sockets");

  if (node_agent_router_executor_socket_)
    node_agent_router_executor_socket_->close();
  if (node_agent_router_handler_socket_)
    node_agent_router_handler_socket_->close();
  if (zmq_context_) zmq_context_->close();

  LOG_DEBUG("Closed ZMQ sockets successfully");
}

void Coordinator::StartHandlerLoop() {
  LOG_DEBUG("Starting handler loop");

  handler_thread_ = std::thread(SETU_LAUNCH_THREAD(
      [this]() { this->HandlerLoop(); }, "CoordinatorHandlerThread"));
}

void Coordinator::StopHandlerLoop() {
  LOG_DEBUG("Stopping handler loop");

  handler_running_ = false;

  if (handler_thread_.joinable()) {
    handler_thread_.join();
  }

  LOG_DEBUG("Handler loop stopped");
}

void Coordinator::StartExecutorLoop() {
  LOG_DEBUG("Starting executor loop");

  executor_thread_ = std::thread(SETU_LAUNCH_THREAD(
      [this]() { this->ExecutorLoop(); }, "CoordinatorExecutorThread"));
}

void Coordinator::StopExecutorLoop() {
  LOG_DEBUG("Stopping executor loop");

  executor_running_ = false;

  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }

  LOG_DEBUG("Executor loop stopped");
}

void Coordinator::HandlerLoop() {
  LOG_DEBUG("Entering handler loop");

  handler_running_ = true;
  while (handler_running_) {
    auto [node_agent_identity, request] =
        SetuCommHelper::RecvWithIdentity<NodeAgentRequest, false>(
            node_agent_router_handler_socket_);
    std::visit(
        [&](const auto& req) {
          HandleNodeAgentRequest(node_agent_identity, req);
        },
        request);
  }
}

void Coordinator::HandleNodeAgentRequest(
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

  // Send response to client
  RegisterTensorShardResponse response(request.request_id, ErrorCode::kSuccess,
                                       shard_ref);
  SetuCommHelper::SendWithIdentity<CoordinatorMessage, false>(
      node_agent_router_handler_socket_, node_agent_identity, response);

  // Send AllocateTensorRequest to NodeAgent to allocate the tensor
  TensorShardIdentifier tensor_shard_id(request.tensor_shard_spec.name,
                                        shard_id);
  AllocateTensorRequest allocate_request(tensor_shard_id);
  SetuCommHelper::SendWithIdentity<CoordinatorMessage, false>(
      node_agent_router_handler_socket_, node_agent_identity, allocate_request);

  LOG_INFO("Sent AllocateTensorRequest for tensor: {}", tensor_shard_id);
}

void Coordinator::HandleNodeAgentRequest(const Identity& node_agent_identity,
                                         const SubmitCopyRequest& request) {
  LOG_INFO("Coordinator received SubmitCopyRequest from {} to {}",
           request.copy_spec.src_name, request.copy_spec.dst_name);

  // TODO: Actually submit the copy operation
  // For now, just log and respond with success
  LOG_INFO("Submitted copy operation: {} -> {} (stub implementation)",
           request.copy_spec.src_name, request.copy_spec.dst_name);

  SubmitCopyResponse response(RequestId(), ErrorCode::kSuccess);
  SetuCommHelper::SendWithIdentity<CoordinatorMessage, false>(
      node_agent_router_handler_socket_, node_agent_identity, response);
}

void Coordinator::HandleNodeAgentRequest(const Identity& node_agent_identity,
                                         const WaitForCopyRequest& request) {
  LOG_INFO("Coordinator received WaitForCopyRequest for copy operation ID: {}",
           request.copy_operation_id);

  // TODO: Actually wait for the copy operation
  // For now, just log and respond with success
  LOG_INFO("WaitForCopy: {} (stub implementation)", request.copy_operation_id);

  WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);
  SetuCommHelper::SendWithIdentity<CoordinatorMessage, false>(
      node_agent_router_handler_socket_, node_agent_identity, response);
}

void Coordinator::ExecutorLoop() {
  LOG_DEBUG("Entering executor loop");

  executor_running_ = true;
  while (executor_running_) {
    // TODO: Implement executor loop to dispatch plans to NodeAgents
    std::this_thread::sleep_for(kHandleLoopSleepMs);
  }
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
