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
#include "commons/utils/SetuCommHelper.h"
#include "commons/utils/ZmqHelper.h"
//==============================================================================
namespace setu::client {
//==============================================================================
using setu::commons::messages::AnyClientRequest;
using setu::commons::messages::RegisterTensorShardRequest;
using setu::commons::messages::RegisterTensorShardResponse;
using setu::commons::messages::SubmitCopyRequest;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::WaitForCopyRequest;
using setu::commons::messages::WaitForCopyResponse;
using setu::commons::utils::SetuCommHelper;
using setu::commons::utils::ZmqHelper;
//==============================================================================
Client::Client(ClientRank client_rank) : client_rank_(client_rank) {
  zmq_context_ = std::make_shared<zmq::context_t>();
}

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

  LOG_DEBUG("Client connecting to {}", endpoint);

  request_socket_ = ZmqHelper::CreateAndConnectSocket(
      zmq_context_, zmq::socket_type::req, endpoint);

  endpoint_ = endpoint;
  is_connected_ = true;

  LOG_DEBUG("Client connected to {} successfully", endpoint_);
}

void Client::Disconnect() {
  ASSERT_VALID_RUNTIME(is_connected_, "Client is not connected");

  LOG_DEBUG("Client disconnecting from {}", endpoint_);

  if (request_socket_) {
    request_socket_->close();
    request_socket_.reset();
  }

  endpoint_.clear();
  is_connected_ = false;

  LOG_DEBUG("Client disconnected successfully");
}

bool Client::IsConnected() const { return is_connected_; }

const std::string& Client::GetEndpoint() const { return endpoint_; }

std::optional<TensorShardRef> Client::RegisterTensorShard(
    const TensorShardSpec& shard_spec) {
  LOG_DEBUG("Client registering tensor shard: {}", shard_spec.name);

  AnyClientRequest request = RegisterTensorShardRequest(shard_spec);
  SetuCommHelper::Send(request_socket_, request);

  auto response =
      SetuCommHelper::Recv<RegisterTensorShardResponse>(request_socket_);

  LOG_DEBUG("Client received response for tensor shard: {} with error code: {}",
            shard_spec.name, response.error_code);

  if (response.error_code != ErrorCode::kSuccess) {
    return std::nullopt;
  }

  return response.shard_ref;
}

std::optional<CopyOperationId> Client::SubmitCopy(const CopySpec& copy_spec) {
  LOG_DEBUG("Client submitting copy operation from {} to {}",
            copy_spec.src_name, copy_spec.dst_name);

  AnyClientRequest request = SubmitCopyRequest(copy_spec);
  SetuCommHelper::Send(request_socket_, request);

  auto response = SetuCommHelper::Recv<SubmitCopyResponse>(request_socket_);

  LOG_DEBUG("Client received copy operation ID: {}",
            response.copy_operation_id);

  if (response.error_code != ErrorCode::kSuccess) {
    return std::nullopt;
  }

  return response.copy_operation_id;
}

void Client::WaitForCopy(CopyOperationId copy_op_id) {
  LOG_DEBUG("Client waiting for copy operation ID: {}", copy_op_id);

  AnyClientRequest request = WaitForCopyRequest(copy_op_id);
  SetuCommHelper::Send(request_socket_, request);

  auto response = SetuCommHelper::Recv<WaitForCopyResponse>(request_socket_);

  LOG_DEBUG(
      "Client finished waiting for copy operation ID: {} with error code: {}",
      copy_op_id, response.error_code);
}
//==============================================================================
}  // namespace setu::client
//==============================================================================
