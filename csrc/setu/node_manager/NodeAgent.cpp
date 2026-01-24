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
#include "commons/utils/SetuCommHelper.h"
//==============================================================================
namespace setu::node_manager {
//==============================================================================
using setu::commons::DeviceRank;
using setu::commons::LocalDeviceRank;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::Device;
using setu::commons::enums::DeviceKind;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::ClientRequest;
using setu::commons::messages::CoordinatorRequest;
using setu::commons::messages::ExecuteProgramRequest;
using setu::commons::messages::ExecuteProgramResponse;
using setu::commons::messages::ExecuteResponse;
using setu::commons::messages::RegisterTensorShardResponse;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::WaitForCopyResponse;
using setu::commons::utils::SetuCommHelper;
using setu::commons::utils::ZmqHelper;
using setu::coordinator::datatypes::Instruction;
using setu::coordinator::datatypes::Program;
//==============================================================================
constexpr std::int32_t kPollTimeoutMs = 100;
//==============================================================================
NodeAgent::NodeAgent(NodeRank node_rank, std::size_t router_port,
                     std::size_t dealer_executor_port,
                     std::size_t dealer_handler_port)
    : node_rank_(node_rank),
      router_port_(router_port),
      dealer_executor_port_(dealer_executor_port),
      dealer_handler_port_(dealer_handler_port) {
  InitZmqSockets();
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

void NodeAgent::AllocateTensor(const TensorName& tensor_id, ShardId shard_id,
                               DeviceRank device) {
  LOG_DEBUG("Allocating tensor shard: tensor_id={}, shard_id={}, device={}",
            tensor_id, boost::uuids::to_string(shard_id), device);
  // TODO: Implement allocation on device
}

void NodeAgent::CopyOperationFinished(CopyOperationId copy_op_id) {
  LOG_DEBUG("Marking copy operation ID: {} as finished", copy_op_id);
}

void NodeAgent::Execute(Plan plan) { LOG_DEBUG("Executing Plan {}", plan); }

void NodeAgent::InitZmqSockets() {
  LOG_DEBUG("Initializing ZMQ sockets");

  zmq_context_ = std::make_shared<zmq::context_t>();

  client_router_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, router_port_);

  std::string executor_endpoint =
      std::format("tcp://localhost:{}", dealer_executor_port_);
  coordinator_dealer_executor_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, executor_endpoint);

  std::string handler_endpoint =
      std::format("tcp://localhost:{}", dealer_handler_port_);
  coordinator_dealer_handler_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, handler_endpoint);

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

  if (client_router_socket_) client_router_socket_->close();
  if (coordinator_dealer_executor_socket_)
    coordinator_dealer_executor_socket_->close();
  if (coordinator_dealer_handler_socket_)
    coordinator_dealer_handler_socket_->close();
  if (zmq_context_) zmq_context_->close();

  LOG_DEBUG("Closed ZMQ sockets successfully");
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
    auto ready = SetuCommHelper::PollForRead(
        {client_router_socket_, coordinator_dealer_handler_socket_},
        kPollTimeoutMs);

    for (const auto& socket : ready) {
      if (socket == client_router_socket_) {
        auto [identity, request] =
            SetuCommHelper::RecvWithIdentity<ClientRequest, true>(socket);
        std::visit([&](const auto& req) { HandleClientRequest(identity, req); },
                   request);
      } else if (socket == coordinator_dealer_handler_socket_) {
        auto request = SetuCommHelper::Recv<CoordinatorRequest>(socket);
        std::visit([&](const auto& req) { HandleCoordinatorRequest(req); },
                   request);
      }
    }
  }
}

void NodeAgent::HandleClientRequest(const Identity& client_identity,
                                    const RegisterTensorShardRequest& request) {
  LOG_DEBUG("Handling RegisterTensorShardRequest for tensor: {}",
            request.tensor_shard_spec.name);

  // Forward request to coordinator (wrapped in variant)
  ClientRequest variant_request = request;
  SetuCommHelper::Send(coordinator_dealer_handler_socket_, variant_request);

  // Wait for response from coordinator
  auto response = SetuCommHelper::Recv<RegisterTensorShardResponse>(
      coordinator_dealer_handler_socket_);

  // Forward response to client
  SetuCommHelper::SendWithIdentity<RegisterTensorShardResponse, true>(
      client_router_socket_, client_identity, response);
}

void NodeAgent::HandleClientRequest(const Identity& client_identity,
                                    const SubmitCopyRequest& request) {
  LOG_DEBUG("Handling SubmitCopyRequest from {} to {}",
            request.copy_spec.src_name, request.copy_spec.dst_name);

  // Forward request to coordinator (wrapped in variant)
  ClientRequest variant_request = request;
  SetuCommHelper::Send(coordinator_dealer_handler_socket_, variant_request);

  // Wait for response from coordinator
  auto response = SetuCommHelper::Recv<SubmitCopyResponse>(
      coordinator_dealer_handler_socket_);

  // Forward response to client
  SetuCommHelper::SendWithIdentity<SubmitCopyResponse, true>(
      client_router_socket_, client_identity, response);
}

void NodeAgent::HandleClientRequest(const Identity& client_identity,
                                    const WaitForCopyRequest& request) {
  LOG_DEBUG("Handling WaitForCopyRequest for copy operation ID: {}",
            request.copy_operation_id);

  WaitForCopy(request.copy_operation_id);

  WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);

  SetuCommHelper::SendWithIdentity<WaitForCopyResponse, true>(
      client_router_socket_, client_identity, response);
}

void NodeAgent::HandleCoordinatorRequest(const AllocateTensorRequest& request) {
  LOG_DEBUG("Handling AllocateTensorRequest for request: {}", request);
  AllocateTensor(request.tensor_id, request.shard_id, request.device);
}

void NodeAgent::HandleCoordinatorRequest(
    const CopyOperationFinishedRequest& request) {
  LOG_DEBUG("Handling CopyOperationFinishedRequest for request: {}", request);

  // Get and remove all clients waiting for this copy operation
  auto it = pending_waits_.find(request.copy_operation_id);
  if (it != pending_waits_.end()) {
    for (const auto& client_id : it->second) {
      WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);
      SetuCommHelper::SendWithIdentity<WaitForCopyResponse, true>(
          client_router_socket_, client_id, response);
    }
    pending_waits_.erase(it);
  }

  CopyOperationFinished(request.copy_operation_id);
}

void NodeAgent::HandleCoordinatorRequest(const ExecuteRequest& request) {
  LOG_DEBUG("Handling ExecuteRequest for request: {}", request);

  // Put (copy_op_id, node_plan) into executor queue
  executor_queue_.push(std::make_pair(request.copy_op_id, request.node_plan));
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
    for (const auto& [device_rank, program] : plan.worker_programs) {
      // Ensure worker is ready before sending
      EnsureWorkerIsReady(device_rank);

      auto it = workers_req_sockets_.find(device_rank);
      ASSERT_VALID_RUNTIME(it != workers_req_sockets_.end(),
                           "No socket found for device_rank: {}", device_rank);

      // Send ExecuteProgramRequest to worker
      LOG_DEBUG("Sending program with {} instructions to worker {}",
                program.instrs.size(), device_rank);
      ExecuteProgramRequest request(program);
      SetuCommHelper::Send(it->second, request);

      // Wait for acknowledgment from worker
      auto response = SetuCommHelper::Recv<ExecuteProgramResponse>(it->second);
      LOG_DEBUG("Received acknowledgment from worker {}: {}", device_rank,
                response);
    }

    LOG_DEBUG("All workers completed execution for copy_op_id: {}", copy_op_id);

    // Notify coordinator that execution is complete
    ExecuteResponse response(RequestId{}, ErrorCode::kSuccess);
    SetuCommHelper::Send(coordinator_dealer_executor_socket_, response);
  }
}

void NodeAgent::EnsureWorkerIsReady(DeviceRank device_rank) {
  auto it = workers_.find(device_rank);

  if (it == workers_.end()) {
    // Worker doesn't exist, create and start it
    LOG_DEBUG("Creating new worker for device_rank: {}", device_rank);
    Device device = CreateDeviceForRank(device_rank);
    std::size_t worker_port = router_port_ + device_rank + 1;
    auto worker = std::make_unique<Worker>(device, worker_port);
    worker->Start();

    // Create REQ socket to communicate with the worker
    std::string worker_endpoint =
        std::format("tcp://localhost:{}", worker_port);
    ZmqSocketPtr req_socket = ZmqHelper::CreateAndConnectSocket(
        zmq_context_, zmq::socket_type::req, worker_endpoint);
    workers_req_sockets_[device_rank] = std::move(req_socket);

    workers_[device_rank] = std::move(worker);
    LOG_DEBUG("Worker for device_rank {} created and started", device_rank);
  } else if (!it->second->IsRunning()) {
    // Worker exists but is not running, restart it
    LOG_DEBUG("Restarting worker for device_rank: {}", device_rank);
    it->second->Start();
    LOG_DEBUG("Worker for device_rank {} restarted", device_rank);
  }
}

Device NodeAgent::CreateDeviceForRank(DeviceRank device_rank) const {
  // TODO: Make device kind configurable or detect from system
  return Device(DeviceKind::kCuda, node_rank_, device_rank,
                static_cast<LocalDeviceRank>(device_rank));
}

//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
