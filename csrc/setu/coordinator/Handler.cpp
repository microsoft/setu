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
#include "coordinator/Handler.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/QueueUtils.h"
#include "commons/utils/ThreadingUtils.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::NodeId;
using setu::commons::StringToUUID;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::AllocateTensorRequest;
using setu::commons::messages::CopyOperationFinishedRequest;
using setu::commons::messages::DeregisterShardsResponse;
using setu::commons::messages::GetTensorSpecResponse;
using setu::commons::messages::RegisterTensorShardCoordinatorResponse;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::utils::AggregationParticipant;
//==============================================================================
Handler::Handler(Queue<InboxMessage>& inbox_queue,
                 Queue<OutboxMessage>& outbox_queue, MetaStore& metastore,
                 Queue<PlannerTask>& planner_queue,
                 OutboxNotifyFn outbox_notify)
    : inbox_queue_(inbox_queue),
      outbox_queue_(outbox_queue),
      metastore_(metastore),
      planner_queue_(planner_queue),
      outbox_notify_(std::move(outbox_notify)) {}

void Handler::PushOutbox(OutboxMessage msg) {
  outbox_queue_.push(std::move(msg));
  outbox_notify_();
}

void Handler::Start() {
  if (running_.load()) {
    return;
  }
  thread_ = std::thread(SETU_LAUNCH_THREAD([this]() { this->Loop(); },
                                           "CoordinatorHandlerThread"));
}

void Handler::Stop() {
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
}

void Handler::Loop() {
  running_ = true;
  while (running_) {
    try {
      InboxMessage inbox_msg = inbox_queue_.pull();
      std::visit(
          [&](const auto& msg) {
            using T = std::decay_t<decltype(msg)>;
            if constexpr (std::is_same_v<T, RegisterTensorShardRequest>) {
              HandleRegisterTensorShardRequest(inbox_msg.node_agent_identity,
                                               msg);
            } else if constexpr (std::is_same_v<T, SubmitCopyRequest>) {
              HandleSubmitCopyRequest(inbox_msg.node_agent_identity, msg);
            } else if constexpr (std::is_same_v<T, SubmitPullRequest>) {
              HandleSubmitPullRequest(inbox_msg.node_agent_identity, msg);
            } else if constexpr (std::is_same_v<T, ExecuteResponse>) {
              HandleExecuteResponse(inbox_msg.node_agent_identity, msg);
            } else if constexpr (std::is_same_v<T, GetTensorSpecRequest>) {
              HandleGetTensorSpecRequest(inbox_msg.node_agent_identity, msg);
            } else if constexpr (std::is_same_v<T, DeregisterShardsRequest>) {
              HandleDeregisterShardsRequest(inbox_msg.node_agent_identity, msg);
            } else {
              LOG_WARNING("Handler: Unknown message type (index={})",
                          inbox_msg.request.index());
            }
          },
          inbox_msg.request);
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}

void Handler::HandleRegisterTensorShardRequest(
    const Identity& node_agent_identity,
    const RegisterTensorShardRequest& request) {
  LOG_INFO("Coordinator received RegisterTensorShardRequest for tensor: {}",
           request.tensor_shard_spec.name);

  // Parse NodeId from the identity (NodeAgent REQ identity is
  // "uuid_req")
  auto underscore_pos = node_agent_identity.rfind('_');
  ASSERT_VALID_RUNTIME(underscore_pos != std::string::npos,
                       "Invalid node agent identity format: {}",
                       node_agent_identity);
  NodeId owner_node_id =
      StringToUUID(node_agent_identity.substr(0, underscore_pos));

  // Reject registration if the tensor is being deregistered
  if (metastore_.IsTensorDeregistered(request.tensor_shard_spec.name)) {
    LOG_WARNING(
        "Rejecting RegisterTensorShardRequest: tensor '{}' has deregistered "
        "shards",
        request.tensor_shard_spec.name);
    RegisterTensorShardCoordinatorResponse response(
        request.request_id, ErrorCode::kTensorDeregistered);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

  // Register the tensor shard in the metastore with owner information
  auto shard_metadata_ptr =
      metastore_.RegisterTensorShard(request.tensor_shard_spec, owner_node_id);

  // Send response with TensorShardMetadata
  if (shard_metadata_ptr) {
    RegisterTensorShardCoordinatorResponse response(
        request.request_id, ErrorCode::kSuccess, *shard_metadata_ptr);
    PushOutbox(OutboxMessage{node_agent_identity, response});
  } else {
    LOG_ERROR("Failed to register tensor shard: {}", request.tensor_shard_spec);
    RegisterTensorShardCoordinatorResponse response(
        request.request_id, ErrorCode::kInvalidArguments);
    PushOutbox(OutboxMessage{node_agent_identity, response});
    return;
  }

  // Check if all shards for this tensor are registered
  if (metastore_.AllShardsRegistered(request.tensor_shard_spec.name)) {
    LOG_INFO(
        "All shards registered for tensor: {}, sending AllocateTensorRequest "
        "to all owners",
        request.tensor_shard_spec.name);

    // Get tensor metadata to find all owner NodeIds
    auto metadata =
        metastore_.GetTensorMetadata(request.tensor_shard_spec.name);
    ASSERT_VALID_POINTER_ARGUMENT(metadata);

    // Group shard IDs by owner node
    std::unordered_map<NodeId, std::vector<ShardId>> owner_to_shard_ids;
    for (const auto& [shard_id, shard_metadata] : metadata->shards) {
      owner_to_shard_ids[shard_metadata->owner].push_back(shard_id);
    }

    // Send AllocateTensorRequest to each NodeAgent's async (DEALER) socket
    for (const auto& [owner_id, shard_ids] : owner_to_shard_ids) {
      Identity owner_identity = to_string(owner_id) + "_dealer";
      AllocateTensorRequest allocate_request(shard_ids);
      PushOutbox(OutboxMessage{owner_identity, allocate_request});
    }
  }
}

void Handler::HandleSubmitCopyRequest(const Identity& node_agent_identity,
                                      const SubmitCopyRequest& request) {
  LOG_INFO("Coordinator received SubmitCopyRequest from {} to {} for shard {}",
           request.copy_spec.src_name, request.copy_spec.dst_name,
           request.shard_id);

  if (metastore_.IsTensorDeregistered(request.copy_spec.src_name) ||
      metastore_.IsTensorDeregistered(request.copy_spec.dst_name)) {
    LOG_WARNING(
        "Rejecting SubmitCopyRequest: tensor '{}' or '{}' has deregistered "
        "shards",
        request.copy_spec.src_name, request.copy_spec.dst_name);
    SubmitCopyResponse response(request.request_id, CopyOperationId{},
                                ErrorCode::kTensorDeregistered);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

  // Expected = all src shards + all dst shards
  std::size_t expected_shards =
      metastore_.GetNumShardsForTensor(request.copy_spec.src_name) +
      metastore_.GetNumShardsForTensor(request.copy_spec.dst_name);

  HandleShardSubmission(DispatchManager::ShardSubmission{
      request.shard_id, request.copy_spec,
      AggregationParticipant{node_agent_identity, request.request_id},
      expected_shards, std::vector(request.hints), request.hints_fingerprint});
}

void Handler::HandleSubmitPullRequest(const Identity& node_agent_identity,
                                      const SubmitPullRequest& request) {
  LOG_INFO("Coordinator received SubmitPullRequest from {} to {} for shard {}",
           request.copy_spec.src_name, request.copy_spec.dst_name,
           request.shard_id);

  if (metastore_.IsTensorDeregistered(request.copy_spec.src_name) ||
      metastore_.IsTensorDeregistered(request.copy_spec.dst_name)) {
    LOG_WARNING(
        "Rejecting SubmitPullRequest: tensor '{}' or '{}' has deregistered "
        "shards",
        request.copy_spec.src_name, request.copy_spec.dst_name);
    SubmitCopyResponse response(request.request_id, CopyOperationId{},
                                ErrorCode::kTensorDeregistered);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

  // For Pull: expected shards = number of DESTINATION shards only (one-sided)
  std::size_t expected_shards =
      metastore_.GetNumShardsForTensor(request.copy_spec.dst_name);

  HandleShardSubmission(DispatchManager::ShardSubmission{
      request.shard_id, request.copy_spec,
      AggregationParticipant{node_agent_identity, request.request_id},
      expected_shards, std::vector(request.hints), request.hints_fingerprint});
}

void Handler::HandleShardSubmission(
    DispatchManager::ShardSubmission submission) {
  auto result = dispatch_manager_.SubmitShard(
      std::move(submission),
      [](const CopySpec& stored, const CopySpec& incoming) -> bool {
        if (*incoming.src_selection != *stored.src_selection) {
          LOG_ERROR("Shard submission {} -> {}: source selection mismatch",
                    incoming.src_name, incoming.dst_name);
          return false;
        }
        if (*incoming.dst_selection != *stored.dst_selection) {
          LOG_ERROR("Shard submission {} -> {}: destination selection mismatch",
                    incoming.src_name, incoming.dst_name);
          return false;
        }
        return true;
      });

  std::visit(
      [&](auto&& alt) {
        using T = std::decay_t<decltype(alt)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          // Pending — not all shards submitted yet
        } else if constexpr (std::is_same_v<
                                 T, DispatchManager::CancelledAggregation>) {
          for (const auto& participant : alt.participants) {
            LOG_WARNING(
                "Cancelling pending copy submission for participant {} due to "
                "validation error",
                participant.identity);
            SubmitCopyResponse error_response(participant.request_id,
                                              CopyOperationId{},
                                              ErrorCode::kInvalidArguments);
            PushOutbox(OutboxMessage{participant.identity, error_response});
          }
        } else if constexpr (std::is_same_v<
                                 T, DispatchManager::CompletedAggregation>) {
          auto [copy_op_id, state] = dispatch_manager_.FinalizeAggregation(alt);

          LOG_INFO(
              "All shards submitted for {} -> {}, "
              "copy_op_id={}, adding to planner queue",
              alt.spec.src_name, alt.spec.dst_name, copy_op_id);

          planner_queue_.push(PlannerTask{copy_op_id, alt.spec, state,
                                          HintStore(std::move(alt.hints))});

          for (const auto& participant : alt.participants) {
            SubmitCopyResponse response(participant.request_id, copy_op_id,
                                        ErrorCode::kSuccess);
            PushOutbox(OutboxMessage{participant.identity, response});
          }
        }
      },
      std::move(result));
}

void Handler::HandleExecuteResponse(const Identity& /*node_identity*/,
                                    const ExecuteResponse& response) {
  auto completed_state = dispatch_manager_.RecordResponse(response.copy_op_id);
  if (!completed_state.has_value()) return;

  // All participants completed — notify submitters
  const auto& state = *completed_state;
  for (const auto& submitter_identity : state->submitters) {
    CopyOperationFinishedRequest finish_req(response.copy_op_id);
    PushOutbox(OutboxMessage{submitter_identity, finish_req});
  }

  // Check if any deferred deregistrations are now unblocked
  auto unblocked = deregistration_tracker_.Resolve(response.copy_op_id);
  for (auto& dereg : unblocked) {
    LOG_INFO(
        "All blocking copies completed for deregistration from {} — "
        "proceeding with deregistration",
        dereg.node_agent_identity);

    metastore_.DeregisterShards(dereg.shards_by_tensor);

    DeregisterShardsResponse dereg_response(dereg.request_id,
                                            ErrorCode::kSuccess);
    outbox_queue_.push(
        OutboxMessage{dereg.node_agent_identity, dereg_response});
  }
}

void Handler::HandleGetTensorSpecRequest(const Identity& node_agent_identity,
                                         const GetTensorSpecRequest& request) {
  LOG_DEBUG("Coordinator received GetTensorSpecRequest for tensor: {}",
            request.tensor_name);

  if (metastore_.IsTensorDeregistered(request.tensor_name)) {
    LOG_WARNING(
        "Rejecting GetTensorSpecRequest: tensor '{}' has deregistered shards",
        request.tensor_name);
    GetTensorSpecResponse response(request.request_id,
                                   ErrorCode::kTensorDeregistered);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

  const auto* tensor_spec = metastore_.GetTensorSpec(request.tensor_name);
  ASSERT_VALID_RUNTIME(
      tensor_spec != nullptr,
      "TensorSpec must exist for tensor '{}' — at least one shard should have "
      "been registered before GetTensorSpecRequest is sent",
      request.tensor_name);

  GetTensorSpecResponse response(request.request_id, ErrorCode::kSuccess,
                                 *tensor_spec);
  PushOutbox(OutboxMessage{node_agent_identity, response});
}

/// Deregistration is two-phase (see MetaStore class docstring):
///
///   1. Mark. Immediately mark all affected tensors as deregistered so new
///      registrations, copies, and pulls are rejected. Cancel any partial
///      shard aggregation groups that reference these tensors.
///
///   2. Remove. Remove shard metadata from the MetaStore and send a
///      success response to the requesting NodeAgent. If there are
///      in-flight copy operations touching these tensors, defer this step
///      using deregistration_tracker_ (a PendingOperations instance) which
///      releases the deregistration request once all blocking copy
///      operations have finished. On receiving the response, the NodeAgent
///      cleans up its own local state (shard mappings, caches, blocker
///      registrations) and forwards the response to the client.
void Handler::HandleDeregisterShardsRequest(
    const Identity& node_agent_identity,
    const DeregisterShardsRequest& request) {
  LOG_INFO("Coordinator received DeregisterShardsRequest from {}",
           node_agent_identity);

  // Collect tensor names being deregistered
  std::set<TensorName> tensor_names;
  for (const auto& [name, _] : request.shards_by_tensor) {
    tensor_names.insert(name);
  }

  // Mark tensors as deregistered immediately to prevent new copy/pull
  // submissions and registrations from being accepted while deregistration
  // is in progress (even if the actual shard removal is deferred).
  for (const auto& name : tensor_names) {
    metastore_.MarkTensorDeregistered(name);
  }

  // Cancel partial entries in the dispatch manager for these tensors.
  // This cleans up groups that will never complete because the shards are
  // going away.
  auto cancelled_participants =
      dispatch_manager_.CancelPendingByTensors(tensor_names);

  // Send error responses to cancelled participants
  for (const auto& participant : cancelled_participants) {
    LOG_INFO(
        "Cancelling pending copy submission for participant {} due to tensor "
        "deregistration",
        participant.identity);
    SubmitCopyResponse error_response(participant.request_id, CopyOperationId{},
                                      ErrorCode::kTensorDeregistered);
    outbox_queue_.push(OutboxMessage{participant.identity, error_response});
  }

  // Find all in-flight copy operations that involve any of the tensors
  // being deregistered
  auto blocking_ops = dispatch_manager_.FindBlockingOperations(tensor_names);

  PendingDeregistration dereg_data{node_agent_identity, request.request_id,
                                   request.shards_by_tensor};

  if (blocking_ops.empty()) {
    // No in-flight copies — deregister immediately
    metastore_.DeregisterShards(dereg_data.shards_by_tensor);
    DeregisterShardsResponse response(request.request_id, ErrorCode::kSuccess);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
    return;
  }

  auto immediate = deregistration_tracker_.AddWaiter(
      request.request_id, std::move(blocking_ops), std::move(dereg_data));

  if (immediate.has_value()) {
    // All blocking copies already resolved — deregister immediately
    metastore_.DeregisterShards(immediate->shards_by_tensor);
    DeregisterShardsResponse response(request.request_id, ErrorCode::kSuccess);
    outbox_queue_.push(OutboxMessage{node_agent_identity, response});
  } else {
    LOG_INFO(
        "Deferring deregistration for {} tensors from {} — blocked by "
        "in-flight copy operations",
        tensor_names.size(), node_agent_identity);
  }
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
