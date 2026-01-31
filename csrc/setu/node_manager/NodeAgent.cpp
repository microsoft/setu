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
using setu::commons::utils::Comm;
using setu::commons::utils::PrepareTensorIPCSpec;
using setu::commons::utils::ZmqHelper;
using setu::planner::Plan;
//==============================================================================
constexpr std::int32_t kPollTimeoutMs = 100;
//==============================================================================
// NodeAgent Implementation
//==============================================================================
NodeAgent::NodeAgent(NodeId node_id, std::size_t port,
                     std::string coordinator_endpoint,
                     const std::vector<Device>& devices)
    : node_id_(node_id),
      port_(port),
      coordinator_endpoint_(std::move(coordinator_endpoint)),
      devices_(devices),
      zmq_context_(std::make_shared<zmq::context_t>()) {
  handler_ = std::make_unique<Handler>(node_id_, zmq_context_, port_,
                                       coordinator_endpoint_, executor_queue_);
  executor_ = std::make_unique<Executor>(
      node_id_, zmq_context_, coordinator_endpoint_, devices_, executor_queue_);
}

NodeAgent::~NodeAgent() {
  Stop();
  if (zmq_context_) {
    zmq_context_->close();
  }
}

void NodeAgent::Start() {
  LOG_DEBUG("Starting NodeAgent");
  handler_->Start();
  executor_->Start();
}

void NodeAgent::Stop() {
  LOG_DEBUG("Stopping NodeAgent");

  executor_queue_.close();
  handler_->Stop();
  executor_->Stop();
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

void NodeAgent::CopyOperationFinished(CopyOperationId copy_op_id) {
  LOG_DEBUG("Marking copy operation ID: {} as finished", copy_op_id);
}

void NodeAgent::Execute(Plan plan) {
  LOG_DEBUG("Executing Plan {}", plan.ToString());
}

//==============================================================================
// Handler Implementation
//==============================================================================
NodeAgent::Handler::Handler(
    NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
    std::size_t port, const std::string& coordinator_endpoint,
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue)
    : node_id_(node_id),
      zmq_context_(zmq_context),
      port_(port),
      coordinator_endpoint_(coordinator_endpoint),
      executor_queue_(executor_queue) {
  InitSockets();
}

NodeAgent::Handler::~Handler() {
  Stop();
  CloseSockets();
}

void NodeAgent::Handler::InitSockets() {
  LOG_DEBUG("Handler: Initializing ZMQ sockets");

  client_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, port_);

  Identity identity = to_string(node_id_);
  coordinator_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, coordinator_endpoint_, identity);

  LOG_DEBUG("Handler: Initialized ZMQ sockets with identity={}", identity);
}

void NodeAgent::Handler::CloseSockets() {
  LOG_DEBUG("Handler: Closing ZMQ sockets");

  if (client_socket_) {
    client_socket_->close();
  }
  if (coordinator_socket_) {
    coordinator_socket_->close();
  }

  LOG_DEBUG("Handler: Closed ZMQ sockets successfully");
}

void NodeAgent::Handler::Start() {
  if (running_.load()) {
    return;
  }
  LOG_DEBUG("Starting handler loop");
  thread_ = std::thread(
      SETU_LAUNCH_THREAD([this]() { this->Loop(); }, "HandlerLoopThread"));
}

void NodeAgent::Handler::Stop() {
  LOG_DEBUG("Stopping handler loop");
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
  LOG_DEBUG("Handler loop stopped");
}

void NodeAgent::Handler::Loop() {
  LOG_DEBUG("Entering handler loop");

  running_ = true;
  while (running_) {
    auto ready = Comm::PollForRead({client_socket_, coordinator_socket_},
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

void NodeAgent::Handler::HandleClientMessage(const Identity& client_identity,
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

void NodeAgent::Handler::HandleCoordinatorMessage(
    const CoordinatorMessage& message) {
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

void NodeAgent::Handler::HandleRegisterTensorShardRequest(
    const Identity& client_identity,
    const RegisterTensorShardRequest& request) {
  LOG_DEBUG("Handling RegisterTensorShardRequest for tensor: {}",
            request.tensor_shard_spec.name);

  request_id_to_client_identity_[request.request_id] = client_identity;

  // Store the spec so we can allocate the tensor when Coordinator sends
  // AllocateTensorRequest
  tensor_name_to_spec_.emplace(request.tensor_shard_spec.name,
                               request.tensor_shard_spec);

  Comm::Send<NodeAgentRequest>(coordinator_socket_, request);
}

void NodeAgent::Handler::HandleSubmitCopyRequest(
    const Identity& client_identity, const SubmitCopyRequest& request) {
  LOG_DEBUG("Handling SubmitCopyRequest from {} to {}",
            request.copy_spec.src_name, request.copy_spec.dst_name);

  request_id_to_client_identity_[request.request_id] = client_identity;

  Comm::Send<NodeAgentRequest>(coordinator_socket_, request);
}

void NodeAgent::Handler::HandleWaitForCopyRequest(
    const Identity& client_identity, const WaitForCopyRequest& request) {
  LOG_DEBUG("Handling WaitForCopyRequest for copy operation ID: {}",
            request.copy_operation_id);

  pending_waits_[request.copy_operation_id].push_back(client_identity);

  WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);
}

void NodeAgent::Handler::HandleGetTensorHandleRequest(
    const Identity& client_identity, const GetTensorHandleRequest& request) {
  LOG_DEBUG("Handling GetTensorHandleRequest for tensor: {}",
            request.tensor_name);

  auto it = tensor_name_to_tensor_.find(request.tensor_name);
  if (it == tensor_name_to_tensor_.end()) {
    LOG_ERROR("Tensor not found: {}", request.tensor_name);
    GetTensorHandleResponse response(request.request_id,
                                     ErrorCode::kTensorNotFound);
    Comm::SendWithIdentity<GetTensorHandleResponse>(client_socket_,
                                                    client_identity, response);
    return;
  }

  auto tensor_ipc_spec = PrepareTensorIPCSpec(it->second);
  GetTensorHandleResponse response(request.request_id, ErrorCode::kSuccess,
                                   std::move(tensor_ipc_spec));
  Comm::SendWithIdentity<GetTensorHandleResponse>(client_socket_,
                                                  client_identity, response);

  LOG_DEBUG("Sent tensor handle response for tensor: {}", request.tensor_name);
}

void NodeAgent::Handler::HandleAllocateTensorRequest(
    const AllocateTensorRequest& request) {
  LOG_DEBUG("Handling AllocateTensorRequest for request: {}", request);
  AllocateTensor(tensor_name_to_spec_.at(request.tensor_name));
}

void NodeAgent::Handler::HandleCopyOperationFinishedRequest(
    const CopyOperationFinishedRequest& request) {
  LOG_DEBUG("Handling CopyOperationFinishedRequest for request: {}", request);

  // Get and remove all clients waiting for this copy operation
  auto it = pending_waits_.find(request.copy_operation_id);
  if (it != pending_waits_.end()) {
    for (const auto& client_id : it->second) {
      WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);

      // unblock waiting clients
      Comm::SendWithIdentity<WaitForCopyResponse>(client_socket_, client_id,
                                                  response);
    }
    pending_waits_.erase(it);
  }
}

void NodeAgent::Handler::HandleExecuteRequest(const ExecuteRequest& request) {
  LOG_DEBUG("Handling ExecuteRequest for request: {}", request);

  // Put (copy_op_id, node_plan) into executor queue
  executor_queue_.push(std::make_pair(request.copy_op_id, request.node_plan));
}

void NodeAgent::Handler::HandleRegisterTensorShardResponse(
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

void NodeAgent::Handler::HandleSubmitCopyResponse(
    const SubmitCopyResponse& response) {
  auto it = request_id_to_client_identity_.find(response.request_id);
  if (it == request_id_to_client_identity_.end()) {
    LOG_WARNING(
        "Received SubmitCopyResponse for unknown request_id: {}, ignoring",
        response.request_id);
    return;
  }
  const auto& client_identity = it->second;

  Comm::SendWithIdentity<SubmitCopyResponse>(client_socket_, client_identity,
                                             response);

  request_id_to_client_identity_.erase(it);
}

void NodeAgent::Handler::HandleWaitForCopyResponse(
    const WaitForCopyResponse& response) {
  auto it = request_id_to_client_identity_.find(response.request_id);
  if (it == request_id_to_client_identity_.end()) {
    LOG_WARNING(
        "Received WaitForCopyResponse for unknown request_id: {}, ignoring",
        response.request_id);
    return;
  }
  const auto& client_identity = it->second;

  Comm::SendWithIdentity<WaitForCopyResponse>(client_socket_, client_identity,
                                              response);

  request_id_to_client_identity_.erase(it);
}

void NodeAgent::Handler::AllocateTensor(
    const TensorShardSpec& tensor_shard_spec) {
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

//==============================================================================
// Executor Implementation
//==============================================================================
NodeAgent::Executor::Executor(
    NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
    const std::string& coordinator_endpoint, const std::vector<Device>& devices,
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue)
    : node_id_(node_id),
      zmq_context_(zmq_context),
      coordinator_endpoint_(coordinator_endpoint),
      devices_(devices),
      executor_queue_(executor_queue) {
  InitSockets();
}

NodeAgent::Executor::~Executor() {
  Stop();
  CloseSockets();
}

void NodeAgent::Executor::InitSockets() {
  LOG_DEBUG("Executor: Initializing ZMQ sockets");

  Identity identity = to_string(node_id_);
  coordinator_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, coordinator_endpoint_, identity);

  // TODO: Initialize worker sockets based on devices
  LOG_DEBUG("Executor: devices={}", devices_);

  LOG_DEBUG("Executor: Initialized ZMQ sockets with identity={}", identity);
}

void NodeAgent::Executor::CloseSockets() {
  LOG_DEBUG("Executor: Closing ZMQ sockets");

  // Close worker REQ sockets
  for (auto& [device_rank, socket] : worker_sockets_) {
    if (socket) {
      socket->close();
    }
  }
  worker_sockets_.clear();

  if (coordinator_socket_) {
    coordinator_socket_->close();
  }

  LOG_DEBUG("Executor: Closed ZMQ sockets successfully");
}

void NodeAgent::Executor::Start() {
  if (running_.load()) {
    return;
  }
  LOG_DEBUG("Starting executor loop");
  thread_ = std::thread(
      SETU_LAUNCH_THREAD([this]() { this->Loop(); }, "ExecutorLoopThread"));
}

void NodeAgent::Executor::Stop() {
  LOG_DEBUG("Stopping executor loop");
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
  LOG_DEBUG("Executor loop stopped");
}

void NodeAgent::Executor::Loop() {
  LOG_DEBUG("Entering executor loop");

  running_ = true;
  while (running_) {
    // Block until we receive a (copy_op_id, plan) pair from the queue
    try {
      auto [copy_op_id, plan] = executor_queue_.pull();

      LOG_DEBUG("Executor received plan for copy_op_id: {}", copy_op_id);

      // For each worker program in the plan, send it to the corresponding
      // worker
      for (const auto& [participant, program] : plan.program) {
        // Ensure worker is ready before sending
        auto device_rank = participant.device_rank;
        auto it = worker_sockets_.find(device_rank);
        ASSERT_VALID_RUNTIME(it != worker_sockets_.end(),
                             "No socket found for device_rank: {}",
                             device_rank);

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

      LOG_DEBUG("All workers completed execution for copy_op_id: {}",
                copy_op_id);

      // Notify coordinator that execution is complete
      ExecuteResponse response(RequestId{}, ErrorCode::kSuccess);
      Comm::Send(coordinator_socket_, response);
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      LOG_DEBUG("Executor: executor_queue_ closed, exiting");
      return;
    }
  }
}
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
