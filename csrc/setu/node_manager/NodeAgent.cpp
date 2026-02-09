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
#include "node_manager/worker/NCCLWorker.h"
//==============================================================================
namespace setu::node_manager {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorDim;
using setu::commons::datatypes::TensorDimMap;
using setu::commons::datatypes::TensorShard;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorShardRef;
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
using setu::commons::messages::RegisterTensorShardCoordinatorResponse;
using setu::commons::messages::RegisterTensorShardNodeAgentResponse;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::SubmitPullRequest;
using setu::commons::messages::WaitForCopyRequest;
using setu::commons::messages::WaitForCopyResponse;
using setu::commons::messages::WaitForShardAllocationRequest;
using setu::commons::messages::WaitForShardAllocationResponse;
using setu::commons::utils::Comm;
using setu::commons::utils::PrepareTensorIPCSpec;
using setu::commons::utils::ZmqHelper;
using setu::ir::Instruction;
using setu::ir::Program;
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
  // Create workers and connect them via inproc sockets on the shared context
  for (const auto& device : devices_) {
    auto device_rank = device.LocalDeviceIndex();
    auto endpoint =
        std::format("inproc://node_{}_worker_{}", node_id_, device_rank);
    auto worker = std::make_unique<worker::NCCLWorker>(node_id_, device);
    worker->Connect(zmq_context_, endpoint);
    workers_.emplace(device_rank, std::move(worker));
  }

  handler_ = std::make_unique<Handler>(node_id_, zmq_context_, port_,
                                       coordinator_endpoint_, executor_queue_,
                                       shard_id_to_tensor_);
  executor_ = std::make_unique<Executor>(node_id_, zmq_context_,
                                         coordinator_endpoint_, devices_,
                                         executor_queue_, shard_id_to_tensor_);
}

NodeAgent::~NodeAgent() {
  Stop();
  if (zmq_context_) {
    zmq_context_->close();
  }
}

void NodeAgent::Start() {
  LOG_DEBUG("Starting NodeAgent");
  for (auto& [device_rank, worker] : workers_) {
    worker->Start();
  }
  handler_->Start();
  executor_->Start();
}

void NodeAgent::Stop() {
  LOG_DEBUG("Stopping NodeAgent");

  executor_queue_.close();
  handler_->Stop();
  executor_->Stop();
  for (auto& [device_rank, worker] : workers_) {
    worker->Stop();
  }
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
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue,
    TensorShardsConcurrentMap& shard_id_to_tensor)
    : node_id_(node_id),
      zmq_context_(zmq_context),
      port_(port),
      coordinator_endpoint_(coordinator_endpoint),
      executor_queue_(executor_queue),
      shard_id_to_tensor_(shard_id_to_tensor) {
  InitSockets();
}

NodeAgent::Handler::~Handler() {
  Stop();
  CloseSockets();
}

void NodeAgent::Handler::InitSockets() {
  client_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, port_);

  Identity identity = to_string(node_id_);
  coordinator_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, coordinator_endpoint_, identity);
}

void NodeAgent::Handler::CloseSockets() {
  if (client_socket_) {
    client_socket_->close();
  }
  if (coordinator_socket_) {
    coordinator_socket_->close();
  }
}

void NodeAgent::Handler::Start() {
  if (running_.load()) {
    return;
  }
  thread_ = std::thread(
      SETU_LAUNCH_THREAD([this]() { this->Loop(); }, "HandlerLoopThread"));
}

void NodeAgent::Handler::Stop() {
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
}

void NodeAgent::Handler::Loop() {
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
        } else if constexpr (std::is_same_v<T, SubmitPullRequest>) {
          HandleSubmitPullRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, WaitForCopyRequest>) {
          HandleWaitForCopyRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, GetTensorHandleRequest>) {
          HandleGetTensorHandleRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, WaitForShardAllocationRequest>) {
          HandleWaitForShardAllocationRequest(client_identity, msg);
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
        } else if constexpr (std::is_same_v<
                                 T, RegisterTensorShardCoordinatorResponse>) {
          HandleRegisterTensorShardCoordinatorResponse(msg);
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
  request_router_.TrackRequest(request.request_id, client_identity);

  Comm::Send<NodeAgentRequest>(coordinator_socket_, request);
}

void NodeAgent::Handler::HandleSubmitCopyRequest(
    const Identity& client_identity, const SubmitCopyRequest& request) {
  request_router_.TrackRequest(request.request_id, client_identity);

  Comm::Send<NodeAgentRequest>(coordinator_socket_, request);
}

void NodeAgent::Handler::HandleSubmitPullRequest(
    const Identity& client_identity, const SubmitPullRequest& request) {
  request_router_.TrackRequest(request.request_id, client_identity);

  Comm::Send<NodeAgentRequest>(coordinator_socket_, request);
}

void NodeAgent::Handler::HandleWaitForCopyRequest(
    const Identity& client_identity, const WaitForCopyRequest& request) {
  copy_waits_.AddWaiter(request.copy_operation_id, client_identity);

  WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);
}

void NodeAgent::Handler::HandleGetTensorHandleRequest(
    const Identity& client_identity, const GetTensorHandleRequest& request) {
  // TODO: Think how this will change for a general tensor wrapper
  std::optional<setu::commons::utils::TensorIPCSpec> tensor_ipc_spec;

  bool found_metadata = tensor_shard_metadata_map_.find(request.shard_id) !=
                        tensor_shard_metadata_map_.end();

  if (!found_metadata) {
    LOG_ERROR("Shard not found: {}", request.shard_id);
    GetTensorHandleResponse response(request.request_id,
                                     ErrorCode::kTensorNotFound);
    Comm::SendWithIdentity<GetTensorHandleResponse>(client_socket_,
                                                    client_identity, response);
    return;
  }

  bool found_allocated = shard_id_to_tensor_.visit(
      request.shard_id, [&tensor_ipc_spec](const auto& entry) {
        tensor_ipc_spec.emplace(PrepareTensorIPCSpec(entry.second));
      });

  if (!found_allocated) {
    waiting_for_allocation_clients_[request.shard_id].push_back(
        WaitingClient{client_identity, request});
    LOG_DEBUG("Shard not yet allocated, added client to waiting list: {}",
              request.shard_id);
    return;
  }

  GetTensorHandleResponse response(request.request_id, ErrorCode::kSuccess,
                                   std::move(*tensor_ipc_spec));
  Comm::SendWithIdentity<GetTensorHandleResponse>(client_socket_,
                                                  client_identity, response);
}

void NodeAgent::Handler::HandleWaitForShardAllocationRequest(
    const Identity& client_identity,
    const WaitForShardAllocationRequest& request) {
  // Check if shard is already allocated
  bool already_allocated = shard_id_to_tensor_.contains(request.shard_id);

  if (already_allocated) {
    // Shard is already allocated, respond immediately
    WaitForShardAllocationResponse response(request.request_id,
                                            ErrorCode::kSuccess);
    Comm::SendWithIdentity<WaitForShardAllocationResponse>(
        client_socket_, client_identity, response);
    LOG_DEBUG("Shard {} already allocated, responded immediately",
              request.shard_id);
  } else {
    // Shard not yet allocated, add client to pending waits
    waiting_for_allocation_clients_[request.shard_id].push_back(
        WaitingClient{client_identity, request});
    LOG_DEBUG("Shard {} not yet allocated, client added to pending waits",
              request.shard_id);
  }
}

void NodeAgent::Handler::HandleAllocateTensorRequest(
    const AllocateTensorRequest& request) {
  for (const auto& shard_id : request.shard_ids) {
    auto it = tensor_shard_metadata_map_.find(shard_id);
    ASSERT_VALID_RUNTIME(it != tensor_shard_metadata_map_.end(),
                         "No metadata found for shard: {}", shard_id);

    AllocateTensor(*it->second);

    // Check if there are clients waiting for this shard allocation
    auto waiting_it = waiting_for_allocation_clients_.find(shard_id);
    if (waiting_it == waiting_for_allocation_clients_.end()) {
      continue;
    }

    for (const auto& waiting_client : waiting_it->second) {
      HandleClientMessage(waiting_client.client_identity,
                          waiting_client.request);
    }

    waiting_for_allocation_clients_.erase(waiting_it);
    LOG_DEBUG("Unblocked waiting clients for allocated shard: {}", shard_id);
  }
}

void NodeAgent::Handler::HandleCopyOperationFinishedRequest(
    const CopyOperationFinishedRequest& request) {
  // Get and remove all clients waiting for this copy operation
  auto waiters = copy_waits_.DrainWaiters(request.copy_operation_id);
  for (const auto& client_id : waiters) {
    WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);

    // unblock waiting clients
    Comm::SendWithIdentity<WaitForCopyResponse>(client_socket_, client_id,
                                                response);
  }
}

void NodeAgent::Handler::HandleExecuteRequest(const ExecuteRequest& request) {
  executor_queue_.push(std::make_pair(request.copy_op_id, request.node_plan));
}

void NodeAgent::Handler::HandleRegisterTensorShardCoordinatorResponse(
    const RegisterTensorShardCoordinatorResponse& response) {
  auto client_identity = request_router_.ClaimIdentity(response.request_id);
  if (!client_identity.has_value()) {
    LOG_WARNING(
        "Received RegisterTensorShardCoordinatorResponse for unknown "
        "request_id: {}, ignoring",
        response.request_id);
    return;
  }

  // Reconstruct TensorShardRef from TensorShardMetadata
  std::optional<TensorShardRef> shard_ref;
  if (response.shard_metadata.has_value()) {
    const auto& metadata = response.shard_metadata.value();

    // Store the metadata for later allocation
    auto metadata_ptr = std::make_shared<TensorShardMetadata>(metadata);
    tensor_shard_metadata_map_.emplace(metadata.id, metadata_ptr);

    // Build TensorDimMap from the spec's dims
    TensorDimMap dims;
    for (const auto& dim_spec : metadata.spec.dims) {
      dims.emplace(dim_spec.name, TensorDim(dim_spec.name, dim_spec.size));
    }

    shard_ref.emplace(metadata.spec.name, metadata.id, std::move(dims));
  }

  // Send RegisterTensorShardNodeAgentResponse to client
  RegisterTensorShardNodeAgentResponse client_response(
      response.request_id, response.error_code, std::move(shard_ref));
  Comm::SendWithIdentity<RegisterTensorShardNodeAgentResponse>(
      client_socket_, *client_identity, client_response);
}

void NodeAgent::Handler::HandleSubmitCopyResponse(
    const SubmitCopyResponse& response) {
  auto client_identity = request_router_.ClaimIdentity(response.request_id);
  if (!client_identity.has_value()) {
    LOG_WARNING(
        "Received SubmitCopyResponse for unknown request_id: {}, ignoring",
        response.request_id);
    return;
  }

  Comm::SendWithIdentity<SubmitCopyResponse>(client_socket_, *client_identity,
                                             response);
}

void NodeAgent::Handler::HandleWaitForCopyResponse(
    const WaitForCopyResponse& response) {
  auto client_identity = request_router_.ClaimIdentity(response.request_id);
  if (!client_identity.has_value()) {
    LOG_WARNING(
        "Received WaitForCopyResponse for unknown request_id: {}, ignoring",
        response.request_id);
    return;
  }

  Comm::SendWithIdentity<WaitForCopyResponse>(client_socket_, *client_identity,
                                              response);
}

void NodeAgent::Handler::AllocateTensor(
    const TensorShardMetadata& shard_metadata) {
  const auto& spec = shard_metadata.spec;

  // Build the shape from dims (using owned size for each dimension)
  std::vector<std::int64_t> shape;
  shape.reserve(spec.dims.size());
  for (const auto& dim_spec : spec.dims) {
    shape.push_back(static_cast<std::int64_t>(dim_spec.GetOwnedSize()));
  }

  // Create tensor options with dtype and device from spec
  auto options =
      torch::TensorOptions().dtype(spec.dtype).device(spec.device.torch_device);
  torch::Tensor tensor = torch::empty(shape, options);

  shard_id_to_tensor_.insert_or_assign(shard_metadata.id, tensor);

  LOG_DEBUG("Successfully allocated shard {} with shape {} on device {}",
            shard_metadata.id, shape, spec.device.torch_device.str());
}

//==============================================================================
// Executor Implementation
//==============================================================================
NodeAgent::Executor::Executor(
    NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
    const std::string& coordinator_endpoint, const std::vector<Device>& devices,
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue,
    TensorShardsConcurrentMap const& shard_id_to_tensor)
    : node_id_(node_id),
      zmq_context_(zmq_context),
      coordinator_endpoint_(coordinator_endpoint),
      devices_(devices),
      executor_queue_(executor_queue),
      shard_id_to_tensor_(shard_id_to_tensor) {
  InitSockets();
}

NodeAgent::Executor::~Executor() {
  Stop();
  CloseSockets();
}

void NodeAgent::Executor::InitSockets() {
  Identity identity = to_string(node_id_);
  coordinator_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, coordinator_endpoint_,
      identity + "_executor");

  // Connect REQ sockets to each worker's inproc endpoint
  for (const auto& device : devices_) {
    auto device_rank = device.LocalDeviceIndex();
    auto endpoint = std::format("inproc://node_{}_worker_{}", node_id_,
                                device_rank);

    auto socket =
        std::make_shared<zmq::socket_t>(*zmq_context_, zmq::socket_type::req);
    socket->set(zmq::sockopt::linger, 0);
    socket->connect(endpoint);

    worker_sockets_.emplace(device_rank, std::move(socket));
  }
}

void NodeAgent::Executor::CloseSockets() {
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
}

void NodeAgent::Executor::Start() {
  if (running_.load()) {
    return;
  }
  thread_ = std::thread(
      SETU_LAUNCH_THREAD([this]() { this->Loop(); }, "ExecutorLoopThread"));
}

void NodeAgent::Executor::Stop() {
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
}

void NodeAgent::Executor::Loop() {
  running_ = true;
  while (running_) {
    // Block until we receive a (copy_op_id, plan) pair from the queue
    try {
      auto [copy_op_id, plan] = executor_queue_.pull();

      // For each worker program in the plan, send it to the corresponding
      // worker. We send all programs first so workers can execute in parallel,
      // then collect all responses.
      std::vector<std::int32_t> sent_device_ranks;
      for (auto& [participant, program] : plan.program) {
        auto device_rank = participant.LocalDeviceIndex();
        auto it = worker_sockets_.find(device_rank);
        ASSERT_VALID_RUNTIME(it != worker_sockets_.end(),
                             "No socket found for device_rank: {}",
                             device_rank);

        // Populate device ptrs for instructions
        EmbellishProgram(program);

        // Send ExecuteProgramRequest to worker
        LOG_DEBUG("Sending program with {} instructions to worker {}",
                  program.size(), device_rank);
        ExecuteProgramRequest request(program);
        Comm::Send(it->second, request);
        sent_device_ranks.push_back(device_rank);
      }

      // Wait for acknowledgment from all workers
      for (auto device_rank : sent_device_ranks) {
        auto it = worker_sockets_.find(device_rank);
        [[maybe_unused]] auto response =
            Comm::Recv<ExecuteProgramResponse>(it->second);
      }

      LOG_DEBUG("All workers completed execution for copy_op_id: {}",
                copy_op_id);

      // Notify coordinator that execution is complete
      ExecuteResponse response(RequestId{}, copy_op_id, ErrorCode::kSuccess);
      LOG_INFO(
          "NodeAgent Executor: Sending ExecuteResponse to Coordinator for "
          "copy_op_id={}",
          copy_op_id);
      Comm::Send<NodeAgentRequest>(coordinator_socket_, response);
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}

void NodeAgent::Executor::EmbellishProgram(Program& program) {
  auto const DevicePtrLookup = [this](const ShardRef& ref) -> DevicePtr {
    DevicePtr result = nullptr;
    bool found = this->shard_id_to_tensor_.visit(
        ref.shard_id,
        [&result](const auto& entry) { result = entry.second.data_ptr(); });
    ASSERT_VALID_RUNTIME(found,
                         "Embellish failed: Tensor: {}, Shard: {} not found in "
                         "NodeAgent registry.",
                         ref.tensor_name ? *ref.tensor_name : "<unknown>",
                         ref.shard_id);
    return result;
  };

  for (auto& instr : program) {
    instr.Embellish(DevicePtrLookup);
  }
}
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
