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
#include "client/Client.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/Comm.h"
#include "commons/utils/ZmqHelper.h"
#include "messaging/Messages.h"
//==============================================================================
namespace setu::client {
//==============================================================================
using setu::commons::datatypes::TensorSelection;
using setu::commons::messages::ClientRequest;
using setu::commons::messages::DeregisterShardsRequest;
using setu::commons::messages::DeregisterShardsResponse;
using setu::commons::messages::GetTensorHandleRequest;
using setu::commons::messages::GetTensorHandleResponse;
using setu::commons::messages::GetTensorSelectionRequest;
using setu::commons::messages::GetTensorSelectionResponse;
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
using setu::commons::utils::ZmqHelper;
//==============================================================================
Client::Client() { zmq_context_ = std::make_shared<zmq::context_t>(); }

Client::~Client() {
  if (is_connected_) {
    Disconnect();
  }
  if (zmq_context_) {
    zmq_context_->close();
  }
}

void Client::Connect(const std::string& endpoint) {
  ASSERT_VALID_ARGUMENTS(!is_connected_,
                         "Client is already connected to {}. Disconnect first.",
                         endpoint_);
  ASSERT_VALID_ARGUMENTS(!endpoint.empty(), "Endpoint cannot be empty");

  request_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::req, endpoint);

  endpoint_ = endpoint;
  is_connected_ = true;

  LOG_DEBUG("Client connected to {} successfully", endpoint_);
}

void Client::Disconnect() {
  ASSERT_VALID_RUNTIME(is_connected_, "Client is not connected");

  // Deregister all owned shards before disconnecting
  if (!tensor_shards_.empty()) {
    DeregisterShards();
    tensor_shards_.clear();
  }

  if (request_socket_) {
    request_socket_->close();
    request_socket_.reset();
  }

  endpoint_.clear();
  is_connected_ = false;

  LOG_DEBUG("Client disconnected successfully");
}

void Client::DeregisterShards() {
  // Build map of tensor name -> shard IDs from local tracking
  std::unordered_map<TensorName, std::vector<ShardId>> shards_by_tensor;
  for (const auto& [name, refs] : tensor_shards_) {
    std::vector<ShardId> shard_ids;
    shard_ids.reserve(refs.size());
    for (const auto& ref : refs) {
      shard_ids.push_back(ref->shard_id);
    }
    shards_by_tensor.emplace(name, std::move(shard_ids));
  }

  ClientRequest request = DeregisterShardsRequest(std::move(shards_by_tensor));
  Comm::Send(request_socket_, request);

  auto ready = Comm::PollForRead({request_socket_}, kDeregisterTimeoutMs);
  if (ready.empty()) {
    LOG_WARNING(
        "Deregister shards timed out after {}ms, proceeding with disconnect",
        kDeregisterTimeoutMs);
    return;
  }

  auto response = Comm::Recv<DeregisterShardsResponse>(request_socket_);

  LOG_DEBUG("Deregister shards completed with error code: {}",
            response.error_code);
}

bool Client::IsConnected() const { return is_connected_; }

const std::string& Client::GetEndpoint() const { return endpoint_; }

std::optional<TensorShardRef> Client::RegisterTensorShard(
    const TensorShardSpec& shard_spec) {
  ClientRequest request = RegisterTensorShardRequest(shard_spec);
  Comm::Send(request_socket_, request);

  auto response =
      Comm::Recv<RegisterTensorShardNodeAgentResponse>(request_socket_);

  LOG_DEBUG("Client received response for tensor shard: {} with error code: {}",
            shard_spec.name, response.error_code);

  if (response.error_code != ErrorCode::kSuccess) {
    return std::nullopt;
  }

  if (!response.shard_ref.has_value()) {
    LOG_ERROR("Client receieved success response but shard_ref is missing {}",
              shard_spec.name);
    return std::nullopt;
  }

  const auto& shard_ref = response.shard_ref.value();
  tensor_shards_[shard_ref.name].push_back(
      std::make_shared<TensorShardRef>(shard_ref));

  return response.shard_ref;
}

std::optional<CopyOperationId> Client::SubmitCopy(const CopySpec& copy_spec) {
  // Find all shards owned by this client that are involved in the copy
  // (either as source or destination)
  std::vector<ShardId> involved_shards;
  if (auto it = tensor_shards_.find(copy_spec.src_name);
      it != tensor_shards_.end()) {
    for (const auto& shard_ref : it->second) {
      involved_shards.push_back(shard_ref->shard_id);
    }
  }
  if (auto it = tensor_shards_.find(copy_spec.dst_name);
      it != tensor_shards_.end()) {
    for (const auto& shard_ref : it->second) {
      involved_shards.push_back(shard_ref->shard_id);
    }
  }

  ASSERT_VALID_RUNTIME(!involved_shards.empty(),
                       "Client has no shards for src {} or dst {}",
                       copy_spec.src_name, copy_spec.dst_name);

  // Submit a request for each involved shard
  std::optional<CopyOperationId> copy_op_id;
  for (const auto& shard_id : involved_shards) {
    ClientRequest request = SubmitCopyRequest(shard_id, copy_spec);
    Comm::Send(request_socket_, request);

    auto response = Comm::Recv<SubmitCopyResponse>(request_socket_);

    LOG_DEBUG("Client received copy operation ID: {} for shard {}",
              response.copy_operation_id, shard_id);

    if (response.error_code != ErrorCode::kSuccess) {
      return std::nullopt;
    }

    copy_op_id = response.copy_operation_id;
  }

  return copy_op_id;
}

std::optional<CopyOperationId> Client::SubmitPull(const CopySpec& copy_spec) {
  // For Pull: only destination shards submit (one-sided operation)
  auto it = tensor_shards_.find(copy_spec.dst_name);
  ASSERT_VALID_RUNTIME(it != tensor_shards_.end(),
                       "Client has no shards for dst {}", copy_spec.dst_name);

  // Submit a request for each destination shard
  std::optional<CopyOperationId> copy_op_id;
  for (const auto& shard_ref : it->second) {
    const auto shard_id = shard_ref->shard_id;

    ClientRequest request = SubmitPullRequest(shard_id, copy_spec);
    Comm::Send(request_socket_, request);

    auto response = Comm::Recv<SubmitCopyResponse>(request_socket_);

    LOG_DEBUG("Client received pull operation ID: {} for shard {}",
              response.copy_operation_id, shard_id);

    if (response.error_code != ErrorCode::kSuccess) {
      return std::nullopt;
    }

    copy_op_id = response.copy_operation_id;
  }

  return copy_op_id;
}

void Client::WaitForCopy(CopyOperationId copy_op_id) {
  ClientRequest request = WaitForCopyRequest(copy_op_id);
  Comm::Send(request_socket_, request);

  auto response = Comm::Recv<WaitForCopyResponse>(request_socket_);

  LOG_DEBUG(
      "Client finished waiting for copy operation ID: {} with error code: {}",
      copy_op_id, response.error_code);
}

void Client::WaitForShardAllocation(ShardId shard_id) {
  ClientRequest request = WaitForShardAllocationRequest(shard_id);
  Comm::Send(request_socket_, request);

  auto response = Comm::Recv<WaitForShardAllocationResponse>(request_socket_);

  LOG_DEBUG(
      "Client finished waiting for shard allocation: {} with error code: {}",
      shard_id, response.error_code);
}

GetTensorHandleResponse Client::GetTensorHandle(
    const TensorShardRef& shard_ref) {
  ClientRequest request = GetTensorHandleRequest(shard_ref.shard_id);
  Comm::Send(request_socket_, request);

  auto response = Comm::Recv<GetTensorHandleResponse>(request_socket_);

  LOG_DEBUG(
      "Client received tensor handle response for shard: {} with error code: "
      "{}",
      shard_ref.shard_id, response.error_code);

  ASSERT_VALID_RUNTIME(response.error_code == ErrorCode::kSuccess,
                       "Failed to get tensor handle for shard {}: {}",
                       shard_ref.shard_id, response.error_code);
  ASSERT_VALID_RUNTIME(response.tensor_ipc_spec.has_value(),
                       "Tensor IPC spec is missing for shard {}",
                       shard_ref.shard_id);
  ASSERT_VALID_RUNTIME(response.metadata.has_value(),
                       "Metadata is missing for shard {}", shard_ref.shard_id);

  return response;
}

TensorSelectionPtr Client::Select(const TensorName& name) {
  ClientRequest request = GetTensorSelectionRequest(name);
  Comm::Send(request_socket_, request);

  auto response = Comm::Recv<GetTensorSelectionResponse>(request_socket_);

  ASSERT_VALID_RUNTIME(response.error_code == ErrorCode::kSuccess,
                       "Failed to get tensor selection for tensor {}", name);
  ASSERT_VALID_RUNTIME(response.selection.has_value(),
                       "Selection is missing for tensor {}", name);

  return std::make_shared<TensorSelection>(response.selection.value());
}

std::vector<TensorShardRefPtr> Client::GetShards() const {
  std::vector<TensorShardRefPtr> result;
  for (const auto& [name, shards] : tensor_shards_) {
    result.insert(result.end(), shards.begin(), shards.end());
  }
  return result;
}
//==============================================================================
}  // namespace setu::client
//==============================================================================
