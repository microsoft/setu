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
using setu::commons::datatypes::TensorSelection;
using setu::commons::datatypes::TensorShard;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpecPtr;
using setu::commons::enums::DeviceKind;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::ClientRequest;
using setu::commons::messages::CoordinatorMessage;
using setu::commons::messages::CopyOperationFinishedRequest;
using setu::commons::messages::DeregisterShardsRequest;
using setu::commons::messages::DeregisterShardsResponse;
using setu::commons::messages::ExecuteProgramRequest;
using setu::commons::messages::ExecuteProgramResponse;
using setu::commons::messages::ExecuteRequest;
using setu::commons::messages::ExecuteResponse;
using setu::commons::messages::GetTensorHandleRequest;
using setu::commons::messages::GetTensorHandleResponse;
using setu::commons::messages::GetTensorSelectionRequest;
using setu::commons::messages::GetTensorSelectionResponse;
using setu::commons::messages::GetTensorSpecRequest;
using setu::commons::messages::GetTensorSpecResponse;
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
using setu::planner::Plan;
using setu::planner::ir::llc::Instruction;
using setu::planner::ir::llc::Program;
//==============================================================================
constexpr std::int32_t kPollTimeoutMs = 100;
//==============================================================================
// NodeAgent Implementation
//==============================================================================
NodeAgent::NodeAgent(NodeId node_id, std::size_t port,
                     std::string coordinator_endpoint,
                     const std::vector<Device>& devices,
                     std::string lock_base_dir)
    : node_id_(node_id),
      port_(port),
      coordinator_endpoint_(std::move(coordinator_endpoint)),
      devices_(devices),
      zmq_context_(std::make_shared<zmq::context_t>()),
      lock_base_dir_(std::move(lock_base_dir)) {
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
                                       shard_id_to_tensor_, lock_base_dir_);
  // Build register resolver from workers — captures workers_ by reference,
  // safe because NodeAgent outlives the Executor.
  RegisterResolver register_resolver =
      [this](const RegisterRef& ref) -> DevicePtr {
    ASSERT_VALID_RUNTIME(ref.participant.has_value(),
                         "RegisterRef must have a participant for resolution");
    auto device_rank = ref.participant->LocalDeviceIndex();
    auto it = workers_.find(device_rank);
    ASSERT_VALID_RUNTIME(it != workers_.end(), "No worker for device_rank: {}",
                         device_rank);
    return it->second->ResolveRegister(ref);
  };

  executor_ = std::make_unique<Executor>(
      node_id_, zmq_context_, coordinator_endpoint_, devices_, executor_queue_,
      shard_id_to_tensor_, std::move(register_resolver));
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

//==============================================================================
// Handler Implementation
//==============================================================================
NodeAgent::Handler::Handler(
    NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
    std::size_t port, const std::string& coordinator_endpoint,
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue,
    TensorShardsConcurrentMap& shard_id_to_tensor, std::string lock_base_dir)
    : node_id_(node_id),
      zmq_context_(zmq_context),
      port_(port),
      coordinator_endpoint_(coordinator_endpoint),
      executor_queue_(executor_queue),
      shard_id_to_tensor_(shard_id_to_tensor),
      lock_base_dir_(std::move(lock_base_dir)) {
  InitSockets();
}

NodeAgent::Handler::~Handler() {
  Stop();
  CloseSockets();
}

void NodeAgent::Handler::InitSockets() {
  client_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, port_);

  // REQ socket for sync request-response with coordinator
  Identity sync_identity = to_string(node_id_) + "_req";
  sync_socket_ =
      ZmqHelper::CreateAndConnectSocket(zmq_context_, zmq::socket_type::req,
                                        coordinator_endpoint_, sync_identity);

  // DEALER socket for async send/receive with coordinator
  Identity async_identity = to_string(node_id_) + "_dealer";
  async_socket_ =
      ZmqHelper::CreateAndConnectSocket(zmq_context_, zmq::socket_type::dealer,
                                        coordinator_endpoint_, async_identity);
}

void NodeAgent::Handler::CloseSockets() {
  if (client_socket_) {
    client_socket_->close();
  }
  if (sync_socket_) {
    sync_socket_->close();
  }
  if (async_socket_) {
    async_socket_->close();
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
    auto ready =
        Comm::PollForRead({client_socket_, async_socket_}, kPollTimeoutMs);

    for (const auto& socket : ready) {
      if (socket == client_socket_) {
        auto [identity, request] =
            Comm::RecvWithIdentity<ClientRequest>(socket);
        HandleClientMessage(identity, request);
      } else if (socket == async_socket_) {
        auto message = Comm::Recv<CoordinatorMessage>(async_socket_);
        HandleAsyncCoordinatorMessage(message);
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
        } else if constexpr (std::is_same_v<T, GetTensorSelectionRequest>) {
          HandleGetTensorSelectionRequest(client_identity, msg);
        } else if constexpr (std::is_same_v<T, DeregisterShardsRequest>) {
          HandleDeregisterShardsRequest(client_identity, msg);
        }
      },
      request);
}

void NodeAgent::Handler::HandleAsyncCoordinatorMessage(
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
        } else if constexpr (std::is_same_v<T, SubmitCopyResponse>) {
          HandleSubmitCopyResponse(msg);
        } else if constexpr (std::is_same_v<T, DeregisterShardsResponse>) {
          HandleDeregisterShardsResponse(msg);
        }
      },
      message);
}

void NodeAgent::Handler::HandleRegisterTensorShardRequest(
    const Identity& client_identity,
    const RegisterTensorShardRequest& request) {
  // Sync: send via REQ socket, block until coordinator responds
  Comm::Send<NodeAgentRequest>(sync_socket_, request);
  auto coordinator_response = Comm::Recv<CoordinatorMessage>(sync_socket_);

  const auto& resp =
      std::get<RegisterTensorShardCoordinatorResponse>(coordinator_response);

  // Reconstruct TensorShardRef from TensorShardMetadata
  std::optional<TensorShardRef> shard_ref;
  if (resp.shard_metadata.has_value()) {
    const auto& metadata = resp.shard_metadata.value();

    // Store the metadata for later allocation
    auto metadata_ptr = std::make_shared<TensorShardMetadata>(metadata);
    tensor_shard_metadata_map_.emplace(metadata.id, metadata_ptr);

    // Register the shard so clients can wait for its allocation
    pending_shard_allocs_.RegisterBlocker(metadata.id);
    LOG_DEBUG("Registered pending shard allocation for shard: {}", metadata.id);

    // Build TensorDimMap from the spec's dims
    TensorDimMap dims;
    for (const auto& dim_spec : metadata.spec.dims) {
      dims.emplace(dim_spec.name, TensorDim(dim_spec.name, dim_spec.size));
    }

    shard_ref.emplace(metadata.spec.name, metadata.id, std::move(dims));
  }

  // Send RegisterTensorShardNodeAgentResponse to client
  RegisterTensorShardNodeAgentResponse client_response(
      resp.request_id, resp.error_code, std::move(shard_ref));
  Comm::Send<RegisterTensorShardNodeAgentResponse>(
      client_socket_, client_identity, client_response);
}

void NodeAgent::Handler::HandleSubmitCopyRequest(
    const Identity& client_identity, const SubmitCopyRequest& request) {
  request_router_.TrackRequest(request.request_id, client_identity);

  // Async: send via DEALER with delimiter, response comes later
  Comm::Send<NodeAgentRequest>(async_socket_, request);
}

void NodeAgent::Handler::HandleSubmitPullRequest(
    const Identity& client_identity, const SubmitPullRequest& request) {
  request_router_.TrackRequest(request.request_id, client_identity);

  // Async: send via DEALER with delimiter, response comes later
  Comm::Send<NodeAgentRequest>(async_socket_, request);
}

void NodeAgent::Handler::HandleWaitForCopyRequest(
    const Identity& client_identity, const WaitForCopyRequest& request) {
  if (!pending_copies_.IsBlockerRegistered(request.copy_operation_id)) {
    LOG_ERROR("WaitForCopy for unknown copy_operation_id: {}",
              request.copy_operation_id);
    WaitForCopyResponse response(RequestId{}, ErrorCode::kInvalidArguments);
    Comm::Send<WaitForCopyResponse>(client_socket_, client_identity, response);
    return;
  }

  auto immediate = pending_copies_.AddWaiter(Identity{client_identity},
                                             {request.copy_operation_id});

  if (immediate.has_value()) {
    // Already complete (late arrival)
    WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);
    Comm::Send<WaitForCopyResponse>(client_socket_, *immediate, response);
  }
  // Otherwise: stored as pending, will be released on Resolve
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
                                     ErrorCode::kTensorNotFound, std::nullopt,
                                     std::nullopt, lock_base_dir_);
    Comm::Send<GetTensorHandleResponse>(client_socket_, client_identity,
                                        response);
    return;
  }

  bool found_allocated = shard_id_to_tensor_.visit(
      request.shard_id, [&tensor_ipc_spec](const auto& entry) {
        tensor_ipc_spec.emplace(PrepareTensorIPCSpec(entry.second));
      });

  if (!found_allocated) {
    LOG_ERROR("Shard registered but not yet allocated: {}", request.shard_id);
    GetTensorHandleResponse response(request.request_id,
                                     ErrorCode::kTensorNotAllocated);
    Comm::Send<GetTensorHandleResponse>(client_socket_, client_identity,
                                        response);
    return;
  }

  // Look up metadata for this shard
  std::optional<TensorShardMetadata> metadata;
  auto it = tensor_shard_metadata_map_.find(request.shard_id);
  if (it != tensor_shard_metadata_map_.end()) {
    metadata.emplace(*it->second);
  }

  GetTensorHandleResponse response(request.request_id, ErrorCode::kSuccess,
                                   std::move(*tensor_ipc_spec),
                                   std::move(metadata), lock_base_dir_);
  Comm::Send<GetTensorHandleResponse>(client_socket_, client_identity,
                                      response);
}

void NodeAgent::Handler::HandleWaitForShardAllocationRequest(
    const Identity& client_identity,
    const WaitForShardAllocationRequest& request) {
  LOG_DEBUG("WaitForShardAllocation request: shard={}, client={}",
            request.shard_id, client_identity);

  if (!pending_shard_allocs_.IsBlockerRegistered(request.shard_id)) {
    LOG_ERROR("WaitForShardAllocation for unknown shard_id: {}, client={}",
              request.shard_id, client_identity);
    WaitForShardAllocationResponse response(RequestId{},
                                            ErrorCode::kInvalidArguments);
    Comm::Send<WaitForShardAllocationResponse>(client_socket_, client_identity,
                                               response);
    return;
  }

  auto immediate = pending_shard_allocs_.AddWaiter(Identity{client_identity},
                                                   {request.shard_id});

  if (immediate.has_value()) {
    LOG_DEBUG(
        "WaitForShardAllocation: shard {} already complete, responding "
        "immediately to client {}",
        request.shard_id, client_identity);
    WaitForShardAllocationResponse response(RequestId{}, ErrorCode::kSuccess);
    Comm::Send<WaitForShardAllocationResponse>(client_socket_, *immediate,
                                               response);
  } else {
    LOG_DEBUG("WaitForShardAllocation: client {} added as waiter for shard {}",
              client_identity, request.shard_id);
  }
}

void NodeAgent::Handler::HandleGetTensorSelectionRequest(
    const Identity& client_identity, const GetTensorSelectionRequest& request) {
  // Check local cache first
  auto it = tensor_spec_cache_.find(request.tensor_name);
  if (it != tensor_spec_cache_.end()) {
    auto selection = TensorSelection(it->second.name, it->second.dims);
    GetTensorSelectionResponse response(request.request_id, ErrorCode::kSuccess,
                                        selection);
    Comm::Send<GetTensorSelectionResponse>(client_socket_, client_identity,
                                           response);
    return;
  }

  // Sync: send GetTensorSpecRequest via REQ socket, block for response
  GetTensorSpecRequest spec_request(request.tensor_name);
  Comm::Send<NodeAgentRequest>(sync_socket_, spec_request);
  auto coordinator_response = Comm::Recv<CoordinatorMessage>(sync_socket_);

  const auto& spec_response =
      std::get<GetTensorSpecResponse>(coordinator_response);

  if (spec_response.error_code != ErrorCode::kSuccess) {
    LOG_WARNING("GetTensorSpec failed for tensor '{}': {}", request.tensor_name,
                spec_response.error_code);
    GetTensorSelectionResponse error_response(request.request_id,
                                              spec_response.error_code);
    Comm::Send<GetTensorSelectionResponse>(client_socket_, client_identity,
                                           error_response);
    return;
  }

  ASSERT_VALID_RUNTIME(spec_response.tensor_spec.has_value(),
                       "TensorSpec missing in response for tensor {}",
                       request.tensor_name);

  const auto& spec = spec_response.tensor_spec.value();

  // Cache the spec locally for future lookups
  tensor_spec_cache_.emplace(spec.name, spec);

  // Build spanning selection and respond to client
  auto selection = TensorSelection(spec.name, spec.dims);
  GetTensorSelectionResponse response(request.request_id, ErrorCode::kSuccess,
                                      selection);
  Comm::Send<GetTensorSelectionResponse>(client_socket_, client_identity,
                                         response);
}

void NodeAgent::Handler::HandleAllocateTensorRequest(
    const AllocateTensorRequest& request) {
  for (const auto& shard_id : request.shard_ids) {
    auto it = tensor_shard_metadata_map_.find(shard_id);
    ASSERT_VALID_RUNTIME(it != tensor_shard_metadata_map_.end(),
                         "No metadata found for shard: {}", shard_id);

    AllocateTensor(*it->second);

    LOG_DEBUG("Resolving shard allocation: {}", shard_id);
    auto waiters = pending_shard_allocs_.Resolve(shard_id);
    LOG_DEBUG("Released {} waiters for shard {}", waiters.size(), shard_id);
    for (const auto& client_id : waiters) {
      LOG_DEBUG("Responding to waiter {} for shard {}", client_id, shard_id);
      WaitForShardAllocationResponse response(RequestId{}, ErrorCode::kSuccess);
      Comm::Send<WaitForShardAllocationResponse>(client_socket_, client_id,
                                                 response);
    }
  }
}

void NodeAgent::Handler::HandleCopyOperationFinishedRequest(
    const CopyOperationFinishedRequest& request) {
  auto waiters = pending_copies_.Resolve(request.copy_operation_id);
  for (const auto& client_id : waiters) {
    WaitForCopyResponse response(RequestId{}, ErrorCode::kSuccess);
    Comm::Send<WaitForCopyResponse>(client_socket_, client_id, response);
  }
}

void NodeAgent::Handler::HandleExecuteRequest(const ExecuteRequest& request) {
  executor_queue_.push(std::make_pair(request.copy_op_id, request.node_plan));
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

  if (response.error_code == ErrorCode::kSuccess) {
    pending_copies_.RegisterBlocker(response.copy_operation_id);
  }

  Comm::Send<SubmitCopyResponse>(client_socket_, *client_identity, response);
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

void NodeAgent::Handler::HandleDeregisterShardsRequest(
    const Identity& client_identity, const DeregisterShardsRequest& request) {
  LOG_INFO("NodeAgent received DeregisterShardsRequest from client {}",
           client_identity);

  // Track client identity so we can route the async response back
  request_router_.TrackRequest(request.request_id, client_identity);

  // Store the request payload, blocked by the coordinator's response key.
  // The request_id serves as both waiter ID and single blocker key — resolved
  // when the coordinator sends DeregisterShardsResponse with the same
  // request_id.
  (void)pending_deregistrations_.AddWaiter(request.request_id,
                                           {request.request_id},
                                           DeregisterShardsRequest{request});

  Comm::Send<NodeAgentRequest>(async_socket_, request);
}

void NodeAgent::Handler::HandleDeregisterShardsResponse(
    const DeregisterShardsResponse& response) {
  // Resolve the blocker key (request_id) to get the original request payload
  auto original_requests =
      pending_deregistrations_.Resolve(response.request_id);

  // Clean up local state now that the Coordinator has confirmed all pending
  // copies are complete and the shards are deregistered
  for (const auto& original_request : original_requests) {
    for (const auto& [tensor_name, shard_ids] :
         original_request.shards_by_tensor) {
      for (const auto& shard_id : shard_ids) {
        shard_id_to_tensor_.erase(shard_id);
        tensor_shard_metadata_map_.erase(shard_id);
        pending_shard_allocs_.RemoveBlocker(shard_id);

        LOG_DEBUG("Cleaned up shard {} from tensor '{}'", shard_id,
                  tensor_name);
      }
      tensor_spec_cache_.erase(tensor_name);
    }
  }

  // Route response back to the client that initiated the deregistration
  auto client_identity = request_router_.ClaimIdentity(response.request_id);
  if (client_identity.has_value()) {
    DeregisterShardsResponse client_response(response.request_id,
                                             response.error_code);
    Comm::Send<DeregisterShardsResponse>(client_socket_, *client_identity,
                                         client_response);
  } else {
    LOG_WARNING(
        "Received DeregisterShardsResponse for unknown request_id: {}, "
        "ignoring",
        response.request_id);
  }
}

//==============================================================================
// Executor Implementation
//==============================================================================
NodeAgent::Executor::Executor(
    NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
    const std::string& coordinator_endpoint, const std::vector<Device>& devices,
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue,
    TensorShardsConcurrentMap const& shard_id_to_tensor,
    RegisterResolver register_resolver)
    : node_id_(node_id),
      zmq_context_(zmq_context),
      coordinator_endpoint_(coordinator_endpoint),
      devices_(devices),
      executor_queue_(executor_queue),
      shard_id_to_tensor_(shard_id_to_tensor),
      register_resolver_(std::move(register_resolver)) {
  InitSockets();
}

NodeAgent::Executor::~Executor() {
  Stop();
  CloseSockets();
}

void NodeAgent::Executor::InitSockets() {
  Identity identity = to_string(node_id_) + "_executor";
  async_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, coordinator_endpoint_, identity);

  // Connect REQ sockets to each worker's inproc endpoint
  for (const auto& device : devices_) {
    auto device_rank = device.LocalDeviceIndex();
    auto endpoint =
        std::format("inproc://node_{}_worker_{}", node_id_, device_rank);

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

  if (async_socket_) {
    async_socket_->close();
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

        LOG_DEBUG("Embellished program: {} {}", participant, program);

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

      // Notify coordinator that execution is complete (async, fire-and-forget)
      ExecuteResponse response(RequestId{}, copy_op_id, ErrorCode::kSuccess);
      LOG_INFO(
          "NodeAgent Executor: Sending ExecuteResponse to Coordinator for "
          "copy_op_id={}",
          copy_op_id);
      Comm::Send<NodeAgentRequest>(async_socket_, response);
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}

void NodeAgent::Executor::EmbellishProgram(Program& program) {
  auto const resolver = [this](const BufferRef& ref) -> DevicePtr {
    if (ref.IsShard()) {
      const auto& shard = ref.AsShard();
      DevicePtr result = nullptr;
      bool found = this->shard_id_to_tensor_.visit(
          shard.shard_id,
          [&result](const auto& entry) { result = entry.second.data_ptr(); });
      ASSERT_VALID_RUNTIME(
          found,
          "Embellish failed: Tensor: {}, Shard: {} not found in "
          "NodeAgent registry.",
          shard.tensor_name ? *shard.tensor_name : "<unknown>", shard.shard_id);
      return result;
    }
    return register_resolver_(ref.AsRegister());
  };

  for (auto& instr : program) {
    instr.Embellish(resolver);
  }
}
//==============================================================================
std::string NodeAgent::GetDefaultLockBaseDir() {
  auto base = std::filesystem::temp_directory_path() / "setu" / "locks";
  return base.string();
}
//==============================================================================
}  // namespace setu::node_manager
//==============================================================================
