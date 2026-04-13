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
#include "commons/datatypes/TensorShardHandle.h"
#include "commons/utils/CUDAUtils.h"
#include "commons/utils/Comm.h"
#include "commons/utils/EnvUtils.h"
#include "commons/utils/TorchTensorIPC.h"
#include "node_manager/worker/NCCLWorker.h"
#include "planner/RegisterSet.h"
#include "planner/ir/llc/ShardAccess.h"
#include "telemetry/MetricsSink.h"
//==============================================================================
#include <cuda_profiler_api.h>
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
using setu::commons::datatypes::TensorShardPtr;
using setu::commons::datatypes::TensorShardReadHandle;
using setu::commons::datatypes::TensorShardReadHandlePtr;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpecPtr;
using setu::commons::datatypes::TensorShardWriteHandle;
using setu::commons::datatypes::TensorShardWriteHandlePtr;
using setu::commons::enums::DeviceKind;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::ClientRequest;
using setu::commons::messages::ConnectRequest;
using setu::commons::messages::ConnectResponse;
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
using setu::commons::messages::OnboardNodeAgentRequest;
using setu::commons::messages::OnboardNodeAgentResponse;
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
using setu::commons::utils::GetEnv;
using setu::commons::utils::PrepareTensorIPCSpec;
using setu::commons::utils::ZmqHelper;
using setu::commons::utils::ring::CompletionEntry;
using setu::commons::utils::ring::CompletionRingProducer;
using setu::commons::utils::ring::ShmRing;
using setu::planner::Plan;
using setu::planner::RegisterSet;
using setu::planner::ir::llc::GetShardAccess;
using setu::planner::ir::llc::Instruction;
using setu::planner::ir::llc::Program;
using setu::planner::ir::llc::ShardAccessMap;
using setu::planner::ir::llc::ShardAccessMode;
//==============================================================================
constexpr std::int32_t kPollTimeoutMs = 100;
//==============================================================================
// NodeAgent Implementation
//==============================================================================
NodeAgent::NodeAgent(NodeId node_id, std::size_t port,
                     std::string coordinator_endpoint,
                     const std::vector<Device>& devices,
                     std::string lock_base_dir, std::string metrics_endpoint,
                     std::size_t register_size)
    : node_id_(node_id),
      port_(port),
      coordinator_endpoint_(std::move(coordinator_endpoint)),
      devices_(devices),
      zmq_context_(std::make_shared<zmq::context_t>()),
      lock_base_dir_(std::move(lock_base_dir)),
      metrics_endpoint_(std::move(metrics_endpoint)),
      register_size_(register_size) {
  // Create per-worker input queues
  for (const auto& device : devices_) {
    auto device_rank = device.LocalDeviceIndex();
    async_executor_state_.worker_queues.emplace(
        std::piecewise_construct, std::forward_as_tuple(device_rank),
        std::forward_as_tuple());
  }

  // Create shared event pool for cross-worker SyncPoint/Wait.
  shared_event_pool_ = std::make_unique<worker::SharedEventPool>(
      static_cast<std::int32_t>(devices_.size()));

  // Create workers and bind them to queues
  for (const auto& device : devices_) {
    auto device_rank = device.LocalDeviceIndex();
    auto worker = std::make_unique<worker::NCCLWorker>(
        node_id_, device, *shared_event_pool_,

        RegisterSet::Uniform(1, register_size_));
    worker->Bind(async_executor_state_.worker_queues.at(device_rank),
                 async_executor_state_.completion_queue);

    // Set up telemetry sink for each worker (each gets its own ZMQ socket)
    if (!metrics_endpoint_.empty()) {
      auto sink = std::make_shared<setu::telemetry::MetricsSink>(
          zmq_context_, metrics_endpoint_);
      worker->SetMetricsSink(std::move(sink));
    }

    workers_.emplace(device_rank, std::move(worker));
  }

  // Build per-Participant register sets for coordinator onboarding
  std::unordered_map<setu::planner::Participant, RegisterSet>
      participant_register_sets;
  for (const auto& device : devices_) {
    setu::planner::Participant participant(node_id_, device);
    participant_register_sets.emplace(participant,
                                      RegisterSet::Uniform(1, register_size_));
  }

  // Probe P2P access between all device pairs when peer memcpy is requested.
  OnboardNodeAgentRequest::P2PPairs p2p_pairs;
  if (GetEnv<bool>("SETU_WORKER_USE_PEER_MEMCPY", false)) {
    for (const auto& src : devices_) {
      for (const auto& dst : devices_) {
        if (src == dst) continue;
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(
            &can_access, src.LocalDeviceIndex(), dst.LocalDeviceIndex()));
        if (can_access) {
          p2p_pairs.push_back(
              {src.LocalDeviceIndex(), dst.LocalDeviceIndex()});
          LOG_INFO("NodeAgent {}: P2P access available {} -> {}", node_id_,
                   src.LocalDeviceIndex(), dst.LocalDeviceIndex());
        }
      }
    }
  }

  handler_ = std::make_unique<Handler>(
      node_id_, zmq_context_, port_, coordinator_endpoint_, executor_queue_,
      shard_id_to_tensor_, lock_base_dir_,
      std::move(participant_register_sets), std::move(p2p_pairs),
      async_executor_state_);

  // Build register resolver from workers — captures workers_ by reference,
  // safe because NodeAgent outlives the Dispatcher.
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

  dispatcher_ = std::make_unique<Dispatcher>(
      node_id_, executor_queue_, shard_id_to_tensor_,
      std::move(register_resolver), async_executor_state_);

  poller_ = std::make_unique<Poller>(node_id_, zmq_context_,
                                     coordinator_endpoint_,
                                     async_executor_state_);
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
  dispatcher_->Start();
  poller_->Start();
}

void NodeAgent::Stop() {
  LOG_DEBUG("Stopping NodeAgent");

  // 1. Close the executor queue so Dispatcher exits its pull() loop
  executor_queue_.close();

  // 2. Stop Handler (no more new work)
  handler_->Stop();

  // 3. Stop Dispatcher (it has already exited due to queue close)
  dispatcher_->Stop();

  // 4. Close all worker input queues so workers exit their pull() loops
  for (auto& [rank, queue] : async_executor_state_.worker_queues) {
    queue.close();
  }

  // 5. Stop workers (they exit when input queue closes)
  for (auto& [rank, worker] : workers_) {
    worker->Stop();
  }

  // 6. Close completion queue so Poller exits
  async_executor_state_.completion_queue.close();

  // 7. Stop Poller
  poller_->Stop();
}

//==============================================================================
// Handler Implementation
//==============================================================================
NodeAgent::Handler::Handler(
    NodeId node_id, std::shared_ptr<zmq::context_t> zmq_context,
    std::size_t port, const std::string& coordinator_endpoint,
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue,
    TensorShardsConcurrentMap& shard_id_to_tensor, std::string lock_base_dir,
    std::unordered_map<setu::planner::Participant, setu::planner::RegisterSet>
        register_sets,
    OnboardNodeAgentRequest::P2PPairs p2p_pairs,
    AsyncExecutorState& async_executor_state)
    : node_id_(node_id),
      zmq_context_(zmq_context),
      port_(port),
      coordinator_endpoint_(coordinator_endpoint),
      executor_queue_(executor_queue),
      shard_id_to_tensor_(shard_id_to_tensor),
      lock_base_dir_(std::move(lock_base_dir)),
      register_sets_(std::move(register_sets)),
      p2p_pairs_(std::move(p2p_pairs)),
      async_executor_state_(async_executor_state) {
  InitSockets();
}

NodeAgent::Handler::~Handler() {
  Stop();
  CloseSockets();

  // Clean up all client completion rings
  for (auto& [identity, info] : client_ring_info_) {
    ShmRing::Destroy(info.shm_name, info.mmap_ptr, info.mmap_size);
  }
  client_rings_.clear();
  client_ring_info_.clear();
}

Identity NodeAgent::Handler::ResolveCanonicalClientId(
    const Identity& identity) {
  // Strip _query or _submit suffix to get the canonical client UUID
  static constexpr std::string_view kQuerySuffix = "_query";
  static constexpr std::string_view kSubmitSuffix = "_submit";
  if (identity.size() > kQuerySuffix.size() &&
      identity.compare(identity.size() - kQuerySuffix.size(),
                       kQuerySuffix.size(), kQuerySuffix) == 0) {
    return identity.substr(0, identity.size() - kQuerySuffix.size());
  }
  if (identity.size() > kSubmitSuffix.size() &&
      identity.compare(identity.size() - kSubmitSuffix.size(),
                       kSubmitSuffix.size(), kSubmitSuffix) == 0) {
    return identity.substr(0, identity.size() - kSubmitSuffix.size());
  }
  return identity;
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

  // REP socket for control commands (profiling, etc.)
  const auto control_port = port_ + 1;
  control_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::rep, control_port);
  LOG_INFO("NodeAgent[{}]: control socket bound on port {}",
           node_id_, control_port);
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
  if (control_socket_) {
    control_socket_->close();
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

void NodeAgent::Handler::OnboardWithCoordinator() {
  LOG_INFO("NodeAgent {} onboarding with coordinator ({} devices, {} p2p pairs)",
           node_id_, register_sets_.size(), p2p_pairs_.size());

  OnboardNodeAgentRequest request(std::move(register_sets_),
                                  std::move(p2p_pairs_));
  Comm::Send<NodeAgentRequest>(sync_socket_, request);
  auto coordinator_response = Comm::Recv<CoordinatorMessage>(sync_socket_);

  const auto& resp = std::get<OnboardNodeAgentResponse>(coordinator_response);
  ASSERT_VALID_RUNTIME(
      resp.error_code == setu::commons::enums::ErrorCode::kSuccess,
      "OnboardNodeAgent failed with error_code: {}", resp.error_code);

  LOG_INFO("NodeAgent {} onboarding complete", node_id_);
}

void NodeAgent::Handler::Loop() {
  running_ = true;

  // Onboard with coordinator before entering event loop — sends register
  // sets so the coordinator knows about this node's devices.
  OnboardWithCoordinator();

  while (running_) {
    auto ready = Comm::PollForRead(
        {client_socket_, async_socket_, control_socket_}, kPollTimeoutMs);

    for (const auto& socket : ready) {
      if (socket == client_socket_) {
        auto [identity, request] =
            Comm::RecvWithIdentity<ClientRequest>(socket);
        HandleClientMessage(identity, request);
      } else if (socket == async_socket_) {
        auto message = Comm::Recv<CoordinatorMessage>(async_socket_);
        HandleAsyncCoordinatorMessage(message);
      } else if (socket == control_socket_) {
        HandleControlMessage();
      }
    }
  }
}

void NodeAgent::Handler::HandleControlMessage() {
  // Simple string-based protocol on REP socket
  zmq::message_t msg;
  auto result = control_socket_->recv(msg, zmq::recv_flags::none);
  ASSERT_VALID_RUNTIME(result.has_value(), "Control socket recv failed");

  std::string command(static_cast<char*>(msg.data()), msg.size());
  std::string response;

  if (command == "start_profiling") {
    if (!profiling_active_) {
      cudaProfilerStart();
      profiling_active_ = true;
      LOG_INFO("NodeAgent[{}]: profiling started", node_id_);
    }
    response = "ok";
  } else if (command == "stop_profiling") {
    if (profiling_active_) {
      cudaProfilerStop();
      profiling_active_ = false;
      LOG_INFO("NodeAgent[{}]: profiling stopped", node_id_);
    }
    response = "ok";
  } else if (command == "profiling_status") {
    response = profiling_active_ ? "active" : "inactive";
  } else {
    response = "error:unknown_command";
    LOG_WARNING("NodeAgent[{}]: unknown control command: {}",
                node_id_, command);
  }

  control_socket_->send(zmq::buffer(response), zmq::send_flags::none);
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
        } else if constexpr (std::is_same_v<T, ConnectRequest>) {
          HandleConnectRequest(client_identity, msg);
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
  request_id_to_local_id_[request.request_id] = request.local_id;

  // Async: send via DEALER with delimiter, response comes later
  Comm::Send<NodeAgentRequest>(async_socket_, request);
}

void NodeAgent::Handler::HandleSubmitPullRequest(
    const Identity& client_identity, const SubmitPullRequest& request) {
  request_router_.TrackRequest(request.request_id, client_identity);
  request_id_to_local_id_[request.request_id] = request.local_id;

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
        tensor_ipc_spec.emplace(PrepareTensorIPCSpec(entry.second->tensor));
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
  // Push completion to every client's shared-memory ring
  auto clients_it = copy_op_to_clients_.find(request.copy_operation_id);
  if (clients_it != copy_op_to_clients_.end()) {
    for (const auto& client_local : clients_it->second) {
      CompletionEntry entry{request.copy_operation_id, client_local.local_id,
                            ErrorCode::kSuccess, 0};
      auto ring_it = client_rings_.find(client_local.canonical_client_id);
      if (ring_it != client_rings_.end()) {
        ring_it->second->Push(entry);
      }
    }
    copy_op_to_clients_.erase(clients_it);
  }

  // Resolve pending_copies_ for any legacy ZMQ waiters
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

  // Look up and consume the local_id for this request
  auto local_id_it = request_id_to_local_id_.find(response.request_id);
  ASSERT_VALID_RUNTIME(local_id_it != request_id_to_local_id_.end(),
                       "No local_id tracked for request_id: {}",
                       response.request_id);
  const auto local_id = local_id_it->second;
  request_id_to_local_id_.erase(local_id_it);

  const auto canonical_id = ResolveCanonicalClientId(*client_identity);

  if (response.error_code == ErrorCode::kSuccess) {
    pending_copies_.RegisterBlocker(response.copy_operation_id);
    copy_op_to_clients_[response.copy_operation_id].push_back(
        ClientLocalId{canonical_id, local_id});
  } else {
    // Push error to completion ring so the client can detect it
    auto ring_it = client_rings_.find(canonical_id);
    if (ring_it != client_rings_.end()) {
      CompletionEntry entry{CopyOperationId{}, local_id,
                            response.error_code, 0};
      ring_it->second->Push(entry);
    }
  }
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

  auto shard = std::make_shared<TensorShard>(shard_metadata, std::move(tensor),
                                             lock_base_dir_);
  shard_id_to_tensor_.insert_or_assign(shard_metadata.id, std::move(shard));

  LOG_DEBUG("Successfully allocated shard {} with shape {} on device {}",
            shard_metadata.id, shape, spec.device.torch_device.str());
}

void NodeAgent::Handler::HandleDeregisterShardsRequest(
    const Identity& client_identity, const DeregisterShardsRequest& request) {
  LOG_DEBUG("NodeAgent received DeregisterShardsRequest from client {}",
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

  // Collect all shard IDs being deregistered
  std::vector<ShardId> shard_ids;
  for (const auto& original_request : original_requests) {
    for (const auto& [tensor_name, ids] : original_request.shards_by_tensor) {
      shard_ids.insert(shard_ids.end(), ids.begin(), ids.end());
    }
  }

  // Wait for all in-flight GPU work on these shards to complete before
  // cleaning up. This blocks the Handler thread, but deregister is infrequent
  // and the wait is bounded by current in-flight GPU execution time.
  WaitForShardRefCountZero(shard_ids);

  // Clean up local state now that the Coordinator has confirmed all pending
  // copies are complete and the shards are deregistered
  for (const auto& original_request : original_requests) {
    for (const auto& [tensor_name, shard_ids_for_tensor] :
         original_request.shards_by_tensor) {
      for (const auto& shard_id : shard_ids_for_tensor) {
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

void NodeAgent::Handler::HandleConnectRequest(const Identity& client_identity,
                                              const ConnectRequest& request) {
  const auto canonical_id = ResolveCanonicalClientId(client_identity);
  auto shm_name = ShmRing::GenerateShmName("setu_cring", canonical_id);
  const auto capacity = kCompletionRingCapacity;
  const auto mmap_size = ShmRing::ComputeSize<CompletionEntry>(capacity);

  void* ptr = ShmRing::Create<CompletionEntry>(shm_name, capacity);

  client_rings_[canonical_id] = std::make_unique<CompletionRingProducer>(ptr);
  client_ring_info_[canonical_id] = ClientRingInfo{shm_name, ptr, mmap_size};

  ConnectResponse response(request.request_id, ErrorCode::kSuccess, shm_name,
                           capacity);
  Comm::Send<ConnectResponse>(client_socket_, client_identity, response);

  LOG_DEBUG("HandleConnectRequest: created completion ring '{}' for client {}",
            shm_name, canonical_id);
}

void NodeAgent::Handler::WaitForShardRefCountZero(
    const std::vector<ShardId>& shard_ids) {
  std::unique_lock<std::mutex> lock(async_executor_state_.in_flight_mutex);
  async_executor_state_.shard_ref_zero_cv.wait(lock, [&]() {
    for (const auto& shard_id : shard_ids) {
      auto it = async_executor_state_.shard_locks.find(shard_id);
      if (it != async_executor_state_.shard_locks.end() &&
          it->second.ref_count > 0) {
        return false;
      }
    }
    return true;
  });
}

//==============================================================================
// Dispatcher Implementation
//==============================================================================
NodeAgent::Dispatcher::Dispatcher(
    NodeId node_id,
    Queue<std::pair<CopyOperationId, Plan>>& executor_queue,
    TensorShardsConcurrentMap const& shard_id_to_tensor,
    RegisterResolver register_resolver, AsyncExecutorState& state)
    : node_id_(node_id),
      executor_queue_(executor_queue),
      shard_id_to_tensor_(shard_id_to_tensor),
      register_resolver_(std::move(register_resolver)),
      state_(state) {}

NodeAgent::Dispatcher::~Dispatcher() { Stop(); }

void NodeAgent::Dispatcher::Start() {
  if (running_.load()) {
    return;
  }
  thread_ = std::thread(SETU_LAUNCH_THREAD([this]() { this->Loop(); },
                                            "DispatcherLoopThread"));
}

void NodeAgent::Dispatcher::Stop() {
  running_ = false;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void NodeAgent::Dispatcher::Loop() {
  running_ = true;
  while (running_) {
    try {
      auto [copy_op_id, plan] = executor_queue_.pull();
      auto t_dequeued = std::chrono::steady_clock::now();

      // Embellish all programs (resolve symbolic refs to device pointers)
      for (auto& [participant, program] : plan.program) {
        EmbellishProgram(program);
      }

      // Compute merged shard access map for the entire plan
      ShardAccessMap plan_access_map;
      for (const auto& [participant, program] : plan.program) {
        for (const auto& [shard_id, mode] : GetShardAccess(program)) {
          if (mode == ShardAccessMode::kWrite) {
            plan_access_map[shard_id] = ShardAccessMode::kWrite;
          } else {
            plan_access_map.try_emplace(shard_id, ShardAccessMode::kRead);
          }
        }
      }

      // Register in-flight entry, acquire shard locks, increment ref counts
      {
        std::lock_guard<std::mutex> lock(state_.in_flight_mutex);
        for (const auto& [shard_id, mode] : plan_access_map) {
          auto& shard_lock = state_.shard_locks[shard_id];
          if (shard_lock.ref_count == 0) {
            // First in-flight plan touching this shard — acquire file lock
            TensorShardPtr shard_ptr = nullptr;
            bool found = shard_id_to_tensor_.visit(
                shard_id, [&shard_ptr](const auto& entry) {
                  shard_ptr = entry.second;
                });
            ASSERT_VALID_RUNTIME(
                found, "Dispatcher: shard {} not found in map", shard_id);

            shard_lock.mode = mode;
            if (mode == ShardAccessMode::kWrite) {
              shard_lock.write_handle =
                  std::make_shared<TensorShardWriteHandle>(shard_ptr);
            } else {
              shard_lock.read_handle =
                  std::make_shared<TensorShardReadHandle>(shard_ptr);
            }
          }
          shard_lock.ref_count++;
        }
        auto [emplace_it, inserted] = state_.in_flight.emplace(
            copy_op_id,
            InFlightEntry{
                .remaining_workers =
                    static_cast<std::int32_t>(plan.program.size()),
                .shard_access_map = plan_access_map,
                .dispatched_at = t_dequeued});
        ASSERT_VALID_RUNTIME(
            inserted,
            "Dispatcher: duplicate copy_op_id {} (already in in_flight with "
            "remaining_workers={})",
            copy_op_id, emplace_it->second.remaining_workers);
      }

      // Push programs to per-worker queues
      for (auto& [participant, program] : plan.program) {
        auto device_rank = participant.LocalDeviceIndex();
        auto it = state_.worker_queues.find(device_rank);
        ASSERT_VALID_RUNTIME(it != state_.worker_queues.end(),
                             "No worker queue for device_rank: {}",
                             device_rank);
        it->second.push(
            WorkerTask{copy_op_id, std::move(program),
                       std::chrono::steady_clock::now()});
      }

      auto t_dispatched = std::chrono::steady_clock::now();
      auto to_us = [](auto d) {
        return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
      };
      LOG_DEBUG("Dispatcher: copy_op_id={}, embellish+dispatch={}us, workers={}",
                copy_op_id, to_us(t_dispatched - t_dequeued),
                plan.program.size());

    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}

void NodeAgent::Dispatcher::EmbellishProgram(Program& program) {
  auto const resolver = [this](const BufferRef& ref) -> DevicePtr {
    if (ref.IsShard()) {
      const auto& shard = ref.AsShard();
      DevicePtr result = nullptr;
      bool found = this->shard_id_to_tensor_.visit(
          shard.shard_id, [&result](const auto& entry) {
            result = entry.second->GetDevicePtr();
          });
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
// Poller Implementation
//==============================================================================
NodeAgent::Poller::Poller(NodeId node_id,
                          std::shared_ptr<zmq::context_t> zmq_context,
                          const std::string& coordinator_endpoint,
                          AsyncExecutorState& state)
    : node_id_(node_id),
      zmq_context_(zmq_context),
      coordinator_endpoint_(coordinator_endpoint),
      state_(state) {
  InitSockets();
}

NodeAgent::Poller::~Poller() {
  Stop();
  CloseSockets();
}

void NodeAgent::Poller::InitSockets() {
  Identity identity = to_string(node_id_) + "_poller";
  async_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::dealer, coordinator_endpoint_, identity);
}

void NodeAgent::Poller::CloseSockets() {
  if (async_socket_) {
    async_socket_->close();
  }
}

void NodeAgent::Poller::Start() {
  if (running_.load()) {
    return;
  }
  thread_ = std::thread(
      SETU_LAUNCH_THREAD([this]() { this->Loop(); }, "PollerLoopThread"));
}

void NodeAgent::Poller::Stop() {
  running_ = false;
  if (thread_.joinable()) {
    thread_.join();
  }
}

void NodeAgent::Poller::Loop() {
  running_ = true;
  while (running_) {
    try {
      auto completion = state_.completion_queue.pull();

      CopyOperationId copy_op_id = completion.copy_op_id;
      bool all_done = false;
      ShardAccessMap shards_to_release;
      std::chrono::steady_clock::time_point dispatched_at;

      {
        std::lock_guard<std::mutex> lock(state_.in_flight_mutex);
        auto it = state_.in_flight.find(copy_op_id);
        ASSERT_VALID_RUNTIME(
            it != state_.in_flight.end(),
            "Poller: completion for unknown copy_op_id: {}", copy_op_id);

        it->second.remaining_workers--;
        ASSERT_VALID_RUNTIME(
            it->second.remaining_workers >= 0,
            "Poller: remaining_workers went negative for {}", copy_op_id);

        if (it->second.remaining_workers == 0) {
          all_done = true;
          shards_to_release = std::move(it->second.shard_access_map);
          dispatched_at = it->second.dispatched_at;
          state_.in_flight.erase(it);
        }
      }

      if (all_done) {
        // Decrement shard ref counts and release locks when reaching zero
        {
          std::lock_guard<std::mutex> lock(state_.in_flight_mutex);
          for (const auto& [shard_id, mode] : shards_to_release) {
            auto it = state_.shard_locks.find(shard_id);
            ASSERT_VALID_RUNTIME(
                it != state_.shard_locks.end(),
                "Poller: shard lock not found for shard {}", shard_id);

            auto& shard_lock = it->second;
            shard_lock.ref_count--;
            ASSERT_VALID_RUNTIME(
                shard_lock.ref_count >= 0,
                "Poller: shard ref count went negative for shard {}",
                shard_id);
            if (shard_lock.ref_count == 0) {
              // Release file lock handles (RAII destructor releases the lock)
              shard_lock.read_handle.reset();
              shard_lock.write_handle.reset();
              state_.shard_locks.erase(it);
            }
          }
          state_.shard_ref_zero_cv.notify_all();
        }

        auto t_done = std::chrono::steady_clock::now();
        auto total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                t_done - dispatched_at)
                .count();
        LOG_DEBUG("Poller: copy_op_id={} complete, total_latency={}us",
                  copy_op_id, total_us);

        // Notify coordinator that execution is complete
        ExecuteResponse response(RequestId{}, copy_op_id, ErrorCode::kSuccess);
        Comm::Send<NodeAgentRequest>(async_socket_, response);
      }
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
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
