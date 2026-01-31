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
#include "node_manager/NodeAgent.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/Comm.h"
#include "commons/utils/TorchTensorIPC.h"
//==============================================================================
namespace setu::node_manager {
//==============================================================================
using setu::commons::DeviceRank;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::Device;
using setu::commons::enums::DeviceKind;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::ClientRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::CopyOperationFinishedRequest;
using setu::commons::messages::ExecuteProgramRequest;
using setu::commons::messages::ExecuteProgramResponse;
using setu::commons::messages::ExecuteRequest;
using setu::commons::messages::ExecuteResponse;
using setu::commons::messages::GetTensorHandleRequest;
using setu::commons::messages::GetTensorHandleResponse;
using setu::commons::messages::NodeAgentRequest;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::RegisterTensorShardResponse;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::WaitForCopyRequest;
using setu::commons::messages::WaitForCopyResponse;
using setu::commons::utils::PrepareTensorIPCSpec;
using setu::commons::utils::Comm;
using setu::commons::utils::ZmqHelper;
using setu::planner::Plan;
//==============================================================================
constexpr std::int32_t kPollTimeoutMs = 100;
//==============================================================================
NodeAgent::NodeAgent(NodeId node_id, std::size_t router_port,
                     std::size_t dealer_executor_port,
                     std::size_t dealer_handler_port,
                     const std::vector<Device>& devices)
    : node_id_(node_id),
      router_port_(router_port),
      dealer_executor_port_(dealer_executor_port),
      dealer_handler_port_(dealer_handler_port) {
  InitZmqSockets();
  InitWorkers(devices);
}

NodeAgent::~NodeAgent() {
  Stop();
  CloseZmqSockets();
}

void NodeAgent::Start() {
  LOG_DEBUG("Starting NodeAgent");
  if (!handler_running_.load()) {
    StartHandlerLoop();
  }
  if (!executor_running_.load()) {
    StartExecutorLoop();
  }
}

void NodeAgent::Stop() {
  LOG_DEBUG("Stopping NodeAgent");
  StopHandlerLoop();
  StopExecutorLoop();
}

std::optional<TensorShardRef> NodeAgent::RegisterTensorShard(
    const TensorShardSpec& shard_spec) {
  LOG_DEBUG("Registering tensor shard: {}", shard_spec.name);

  // TODO: Implement
  return std::nullopt;
}

std::optional<CopyOperationId> NodeAgent::SubmitCopy(
    const CopySpec& copy_spec) {
  LOG_DEBUG("Submitting copy operation from {} to {}", copy_spec.src_name,
            copy_spec.dst_name);

  // TODO: Implement copy submission
  return std::nullopt;
}

void NodeAgent::WaitForCopy(CopyOperationId copy_op_id) {
  LOG_DEBUG("Waiting for copy operation ID: {}", copy_op_id);

  // TODO: Implement wait for copy
}

void NodeAgent::AllocateTensor(const TensorShardSpec& tensor_shard_spec) {
  LOG_DEBUG("Allocating tensor shard: tensor_shard_spec={}", tensor_shard_spec);

  // Build the shape from dims (using owned size for each dimension)
  std::vector<std::int64_t> shape;
  shape.reserve(tensor_shard_spec.dims.size());
  for (const auto& dim_spec : tensor_shard_spec.dims) {
    shape.push_back(static_cast<std::int64_t>(dim_spec.GetOwnedSize()));
  }

  // Create tensor options with dtype and device from spec
  auto options = torch::TensorOptions()
                     .dtype(tensor_shard_spec.dtype)
                     .device(tensor_shard_spec.device.torch_device);

  torch::Tensor tensor = torch::empty(shape, options);

  // Store the tensor
  tensor_name_to_tensor_[tensor_shard_spec.name] = std::move(tensor);

  LOG_DEBUG("Successfully allocated tensor '{}' with shape {} on device {}",
            tensor_shard_spec.name, shape,
            tensor_shard_spec.device.torch_device.str());
}

void NodeAgent::CopyOperationFinished(CopyOperationId copy_op_id) {
  LOG_DEBUG("Marking copy operation ID: {} as finished", copy_op_id);
}

void NodeAgent::Execute(Plan plan) {
  LOG_DEBUG("Executing Plan {}", plan.ToString());
}

void NodeAgent::InitZmqSockets() {
  LOG_DEBUG("Initializing ZMQ sockets");

  zmq_context_ = std::make_shared<zmq::context_t>();

  client_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, port_);

  coordinator_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, coordinator_endpoint_);

  LOG_DEBUG("Initialized ZMQ sockets successfully");
}

void NodeAgent::CloseZmqSockets() {
  LOG_DEBUG("Closing ZMQ sockets");

  // Close worker REQ sockets
  for (auto& [device_rank, socket] : workers_req_sockets_) {
    if (socket) {
      socket->close();
    }
  }
  workers_req_sockets_.clear();

  if (client_socket_) client_socket_->close();
  if (coordinator_socket_)
    coordinator_socket_->close();
  if (zmq_context_) zmq_context_->close();

  LOG_DEBUG("Closed ZMQ sockets successfully");
}

void NodeAgent::InitWorkers(const std::vector<Device>& devices) {
  LOG_DEBUG("{}", devices);
}

void NodeAgent::StartHandlerLoop() {
  LOG_DEBUG("Starting handler loop");

  handler_thread_ = std::thread(SETU_LAUNCH_THREAD(
      [this]() { this->HandlerLoop(); }, "HandlerLoopThread"));
}

void NodeAgent::StopHandlerLoop() {
  LOG_DEBUG("Stopping handler loop");

  handler_running_ = false;

  if (handler_thread_.joinable()) {
    handler_thread_.join();
  }

  LOG_DEBUG("Handler loop stopped");
}

void NodeAgent::StartExecutorLoop() {
  LOG_DEBUG("Starting executor loop");

  executor_thread_ = std::thread(SETU_LAUNCH_THREAD(
      [this]() { this->ExecutorLoop(); }, "ExecutorLoopThread"));
}

void NodeAgent::StopExecutorLoop() {
  LOG_DEBUG("Stopping executor loop");

  executor_running_ = false;

  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }

  LOG_DEBUG("Executor loop stopped");
}

void NodeAgent::HandlerLoop() {
  LOG_DEBUG("Entering handler loop");

  handler_running_ = true;
  while (handler_running_) {
    auto ready = Comm::PollForRead(
        {client_socket_, coordinator_socket_},
        kPollTimeoutMs);

    for (const auto& socket : ready) {
      if (socket == client_socket_) {
        auto [identity, request] =
            Comm::RecvWithIdentity<ClientRequest>(socket);
        HandleClientMessage(identity, request);
      } else if (socket == coordinator_socket_) {
        auto message = Comm::Recv<CoordinatorMessage>(socket);
        HandleCoordinatorMessage(message);
      }
    }
  }
}

void NodeAgent::HandleClientMessage(const Identity& client_identity,
                                    const ClientRequest& request) {
  std::visit(
      [&](const auto& msg) {
        using T = std::decay_t<decltype(msg)>;
        if constexpr (std::is_same_v<T, RegisterTensorShardRequest>) {
          HandleRegisterTensorShardRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, SubmitCopyRequest>) {
          HandleSubmitCopyRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, WaitForCopyRequest>) {
          HandleWaitForCopyRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, GetTensorHandleRequest>) {
          HandleGetTensorHandleRequest(client_identity, msg);
        }
      },
      request);
}

void NodeAgent::HandleCoordinatorMessage(const CoordinatorMessage& message) {
  std::visit(
      [&](const auto& msg) {
        using T = std::decay_t<decltype(msg)>;
        if constexpr (std::is_same_v<T, AllocateTensorRequest>) {
          HandleAllocateTensorRequest(msg);
        } else if constexpr (std::is_same_v<T, CopyOperationFinishedRequest>) {
          HandleCopyOperationFinishedRequest(msg);
        } else if constexpr (std::is_same_v<T, ExecuteRequest>) {
          HandleExecuteRequest(msg);
        } else if constexpr (std::is_same_v<T, RegisterTensorShardResponse>) {
          HandleRegisterTensorShardResponse(msg);
        } else if constexpr (std::is_same_v<T, SubmitCopyResponse>) {
          HandleSubmitCopyResponse(msg);
        } else if constexpr (std::is_same_v<T, WaitForCopyResponse>) {
          HandleWaitForCopyResponse(msg);
        }
      },
      message);
}

void NodeAgent::HandleRegisterTensorShardRequest(
    const Identity& client_identity,
    const RegisterTensorShardRequest& request) {
  LOG_DEBUG("Handling RegisterTensorShardRequest for tensor: {}",
            request.tensor_shard_spec.name);

  request_id_to_client_identity_[request.request_id] = client_identity;

  // Store the spec so we can allocate the tensor when Coordinator sends
  // AllocateTensorRequest
  tensor_name_to_spec_.emplace(request.tensor_shard_spec.name,
                               request.tensor_shard_spec);

  Comm::Send<NodeAgentRequest>(coordinator_socket_,
                                         request);
}

void NodeAgent::HandleSubmitCopyRequest(const Identity& client_identity,
                                        const SubmitCopyRequest& request) {
  LOG_DEBUG("Handling SubmitCopyRequest from {} to {}",
            request.copy_spec.src_name, request.copy_spec.dst_name);

  request_id_to_client_identity_[request.request_id] = client_identity;

  Comm::Send<NodeAgentRequest>(coordinator_socket_,
                                         request);
}

void NodeAgent::HandleWaitForCopyRequest(const Identity& client_identity,
                                         const WaitForCopyRequest& request) {
  LOG_DEBUG("Handling WaitForCopyRequest for copy operation ID: {}",
            request.copy_operation_id);

  pending_waits_[request.copy_operation_id].push_back(client_identity);

  WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);
}

void NodeAgent::HandleGetTensorHandleRequest(
    const Identity& client_identity, const GetTensorHandleRequest& request) {
  LOG_DEBUG("Handling GetTensorHandleRequest for tensor: {}",
            request.tensor_name);

  auto it = tensor_name_to_tensor_.find(request.tensor_name);
  if (it == tensor_name_to_tensor_.end()) {
    LOG_ERROR("Tensor not found: {}", request.tensor_name);
    GetTensorHandleResponse response(request.request_id,
                                     ErrorCode::kTensorNotFound);
    Comm::SendWithIdentity<GetTensorHandleResponse>(
        client_socket_, client_identity, response);
    return;
  }

  auto tensor_ipc_spec = PrepareTensorIPCSpec(it->second);
  GetTensorHandleResponse response(request.request_id, ErrorCode::kSuccess,
                                   std::move(tensor_ipc_spec));
  Comm::SendWithIdentity<GetTensorHandleResponse>(
      client_socket_, client_identity, response);

  LOG_DEBUG("Sent tensor handle response for tensor: {}", request.tensor_name);
}

void NodeAgent::HandleAllocateTensorRequest(
    const AllocateTensorRequest& request) {
  LOG_DEBUG("Handling AllocateTensorRequest for request: {}", request);
  AllocateTensor(tensor_name_to_spec_.at(request.tensor_name));
}

void NodeAgent::HandleCopyOperationFinishedRequest(
    const CopyOperationFinishedRequest& request) {
  LOG_DEBUG("Handling CopyOperationFinishedRequest for request: {}", request);

  // Get and remove all clients waiting for this copy operation
  auto it = pending_waits_.find(request.copy_operation_id);
  if (it != pending_waits_.end()) {
    for (const auto& client_id : it->second) {
      WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);

      // unblock waiting clients
      Comm::SendWithIdentity<WaitForCopyResponse>(
          client_socket_, client_id, response);
    }
    pending_waits_.erase(it);
  }
}

void NodeAgent::HandleExecuteRequest(const ExecuteRequest& request) {
  LOG_DEBUG("Handling ExecuteRequest for request: {}", request);

  // Put (copy_op_id, node_plan) into executor queue
  executor_queue_.push(std::make_pair(request.copy_op_id, request.node_plan));
}

void NodeAgent::HandleRegisterTensorShardResponse(
    const RegisterTensorShardResponse& response) {
  auto it = request_id_to_client_identity_.find(response.request_id);
  if (it == request_id_to_client_identity_.end()) {
    LOG_WARNING(
        "Received RegisterTensorShardResponse for unknown request_id: "
        "{}, ignoring",
        response.request_id);
    return;
  }
  const auto& client_identity = it->second;

  Comm::SendWithIdentity<RegisterTensorShardResponse>(
      client_socket_, client_identity, response);

  request_id_to_client_identity_.erase(it);
}

void NodeAgent::HandleSubmitCopyResponse(const SubmitCopyResponse& response) {
  auto it = request_id_to_client_identity_.find(response.request_id);
  if (it == request_id_to_client_identity_.end()) {
    LOG_WARNING(
        "Received SubmitCopyResponse for unknown request_id: {}, ignoring",
        response.request_id);
    return;
  }
  const auto& client_identity = it->second;

  Comm::SendWithIdentity<SubmitCopyResponse>(
      client_socket_, client_identity, response);

  request_id_to_client_identity_.erase(it);
}

void NodeAgent::HandleWaitForCopyResponse(const WaitForCopyResponse& response) {
  auto it = request_id_to_client_identity_.find(response.request_id);
  if (it == request_id_to_client_identity_.end()) {
    LOG_WARNING(
        "Received WaitForCopyResponse for unknown request_id: {}, ignoring",
        response.request_id);
    return;
  }
  const auto& client_identity = it->second;

  Comm::SendWithIdentity<WaitForCopyResponse>(
      client_socket_, client_identity, response);

  request_id_to_client_identity_.erase(it);
}

void NodeAgent::ExecutorLoop() {
  LOG_DEBUG("Entering executor loop");

  executor_running_ = true;
  while (executor_running_) {
    // Block until we receive a (copy_op_id, plan) pair from the queue
    std::pair<CopyOperationId, Plan> queue_item;
    executor_queue_.pull(queue_item);
    auto [copy_op_id, plan] = std::move(queue_item);

    LOG_DEBUG("Executor received plan for copy_op_id: {}", copy_op_id);

    // For each worker program in the plan, send it to the corresponding worker
    for (const auto& [participant, program] : plan.program) {
      // Ensure worker is ready before sending
      auto device_rank = participant.device_rank;
      auto it = workers_req_sockets_.find(device_rank);
      ASSERT_VALID_RUNTIME(it != workers_req_sockets_.end(),
                           "No socket found for device_rank: {}", device_rank);

      // Send ExecuteProgramRequest to worker
      LOG_DEBUG("Sending program with {} instructions to worker {}",
                program.size(), device_rank);
      ExecuteProgramRequest request(program);
      Comm::Send(it->second, request);

      // Wait for acknowledgment from worker
      auto response = Comm::Recv<ExecuteProgramResponse>(it->second);
      LOG_DEBUG("Received acknowledgment from worker {}: {}", device_rank,
                response);
    }

    LOG_DEBUG("All workers completed execution for copy_op_id: {}", copy_op_id);

    // Notify coordinator that execution is complete
    ExecuteResponse response(RequestId{}, ErrorCode::kSuccess);
    Comm::Send(coordinator_socket_, response);
  }
}
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
