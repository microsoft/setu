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
#pragma once
//==============================================================================
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/ShardAggregator.h"
#include "coordinator/Types.h"
#include "planner/hints/HintStore.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::GenerateUUID;
using setu::commons::Identity;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::CopySpec;
//==============================================================================

/// @brief Manages the full lifecycle of copy operations: shard aggregation,
/// operation tracking, and response completion.
///
/// Owns pre-dispatch shard aggregation state and post-dispatch in-flight
/// operation tracking. Handler delegates all copy-operation state management
/// here and handles only message routing and deregistration coordination.
class DispatchManager {
 public:
  using AggregationParticipant = setu::commons::utils::AggregationParticipant;

  /// @brief All parameters needed to submit a shard for aggregation.
  struct ShardSubmission {
    ShardId shard_id;
    CopySpec copy_spec;
    AggregationParticipant participant;
    std::size_t expected_count;
    std::vector<setu::planner::hints::CompilerHint> hints;
    std::uint64_t hints_fingerprint;
  };

  struct CompletedAggregation {
    CopySpec spec;
    std::vector<AggregationParticipant> participants;
    std::vector<setu::planner::hints::CompilerHint> hints;
  };

  struct CancelledAggregation {
    std::vector<AggregationParticipant> participants;
  };

  /// Pending (monostate), completed, or cancelled.
  using SubmitResult =
      std::variant<std::monostate, CompletedAggregation, CancelledAggregation>;

  /// @brief Result of FinalizeAggregation: the generated CopyOperationId and
  /// shared state that is also tracked internally.
  struct TrackedOperation {
    CopyOperationId copy_op_id;
    CopyOperationStatePtr state;
  };

  // --- Shard aggregation (pre-dispatch) ---

  /// @brief Submit a shard for aggregation with per-operation hints.
  ///
  /// First-writer-wins: the first shard submission's hints become
  /// authoritative. Subsequent submissions must have a matching
  /// fingerprint; a mismatch cancels the entire operation and returns
  /// all participants so the caller can send error responses.
  template <typename ValidateFn>
  [[nodiscard]] SubmitResult SubmitShard(ShardSubmission submission /*[in]*/,
                                         ValidateFn validate_fn /*[in]*/) {
    CopyKey key{submission.copy_spec.src_name, submission.copy_spec.dst_name};

    // First-writer-wins hint storage
    auto [it, inserted] = pending_hints_.try_emplace(
        key, PendingHints{std::move(submission.hints),
                          submission.hints_fingerprint});
    if (!inserted && submission.hints_fingerprint != it->second.fingerprint) {
      LOG_ERROR(
          "SPMD hint mismatch for {} -> {}: shard {} sent fingerprint {} "
          "but first submission had {} — cancelling operation",
          submission.copy_spec.src_name, submission.copy_spec.dst_name,
          submission.shard_id, submission.hints_fingerprint,
          it->second.fingerprint);
      return CancelKey(key, std::move(submission.participant));
    }

    // Save before move — Submit consumes participant, but we need to
    // reconstruct it if validation fails.
    auto saved_identity = submission.participant.identity;
    auto saved_request_id = submission.participant.request_id;

    bool valid = true;
    auto result = shard_aggregator_.Submit(
        key, submission.shard_id, submission.copy_spec,
        std::move(submission.participant), submission.expected_count,
        [&](const CopySpec& stored, const CopySpec& incoming) {
          valid = validate_fn(stored, incoming);
          return valid;
        });

    if (!valid) {
      return CancelKey(
          key, {std::move(saved_identity), std::move(saved_request_id)});
    }

    if (!result.has_value()) {
      return std::monostate{};
    }

    auto hints_node = pending_hints_.extract(key);

    return CompletedAggregation{std::move(result->payload),
                                std::move(result->participants),
                                std::move(hints_node.mapped().hints)};
  }

  /// @brief Cancel all pending aggregation groups whose src or dst tensor
  /// is in the given set.
  [[nodiscard]] std::vector<AggregationParticipant> CancelPendingByTensors(
      const std::set<TensorName>& tensor_names /*[in]*/) {
    return CancelPendingIf([&tensor_names](const CopyKey& key) {
      return tensor_names.contains(key.src_name) ||
             tensor_names.contains(key.dst_name);
    });
  }

  // --- Post-aggregation finalization ---

  /// @brief Finalize a completed aggregation: generate a CopyOperationId,
  /// extract submitter identities, create CopyOperationState, and track the
  /// operation internally.
  [[nodiscard]] TrackedOperation FinalizeAggregation(
      const CompletedAggregation& completed /*[in]*/) {
    CopyOperationId copy_op_id = GenerateUUID();

    std::vector<Identity> submitters;
    submitters.reserve(completed.participants.size());
    for (const auto& participant : completed.participants) {
      submitters.push_back(participant.identity);
    }

    auto state = std::make_shared<CopyOperationState>(completed.spec,
                                                      std::move(submitters));

    copy_operations_.emplace(copy_op_id, state);

    return TrackedOperation{copy_op_id, std::move(state)};
  }

  // --- In-flight operation tracking (post-dispatch) ---

  /// @brief Record a completed response. Returns the state if ALL responses
  /// received (caller should notify submitters and clean up), nullopt if
  /// more responses are still expected.
  [[nodiscard]] std::optional<CopyOperationStatePtr> RecordResponse(
      CopyOperationId copy_op_id /*[in]*/) {
    auto it = copy_operations_.find(copy_op_id);
    ASSERT_VALID_RUNTIME(
        it != copy_operations_.end(),
        "RecordResponse for unknown copy_op_id: {} — indicates "
        "double-complete or untracked operation",
        copy_op_id);

    auto& state = it->second;
    state->completed_responses++;

    // Atomic load with acquire ordering to synchronize with Executor's write
    const auto expected =
        state->expected_responses.load(std::memory_order_acquire);

    LOG_DEBUG("ExecuteResponse for copy_op_id {}: {}/{} responses received",
              copy_op_id, state->completed_responses, expected);

    if (state->completed_responses == expected) {
      LOG_INFO(
          "All {} participants completed for copy_op_id {}, notifying {} "
          "submitters",
          expected, copy_op_id, state->submitters.size());

      auto completed_state = std::move(state);
      copy_operations_.erase(it);
      return completed_state;
    }

    return std::nullopt;
  }

  /// @brief Find in-flight operations whose src or dst tensor is in the
  /// given set.
  [[nodiscard]] std::set<CopyOperationId> FindBlockingOperations(
      const std::set<TensorName>& tensor_names /*[in]*/) const {
    std::set<CopyOperationId> blocking_ops;
    for (const auto& [copy_op_id, state] : copy_operations_) {
      if (tensor_names.contains(state->spec.src_name) ||
          tensor_names.contains(state->spec.dst_name)) {
        blocking_ops.insert(copy_op_id);
      }
    }
    return blocking_ops;
  }

 private:
  /// @brief Key for tracking copy operations by (src, dst) tensor pair.
  struct CopyKey {
    TensorName src_name;
    TensorName dst_name;

    bool operator==(const CopyKey& other) const = default;
  };

  /// @brief Hash function for CopyKey.
  struct CopyKeyHash {
    std::size_t operator()(const CopyKey& key) const {
      std::size_t seed = 0;
      boost::hash_combine(seed, key.src_name);
      boost::hash_combine(seed, key.dst_name);
      return seed;
    }
  };

  /// @brief Merged hints + fingerprint for a pending aggregation group.
  struct PendingHints {
    std::vector<setu::planner::hints::CompilerHint> hints;
    std::uint64_t fingerprint;
  };

  SubmitResult CancelKey(
      const CopyKey& key /*[in]*/,
      AggregationParticipant triggering_participant /*[in]*/) {
    auto cancelled = shard_aggregator_.Cancel(key);
    cancelled.push_back(std::move(triggering_participant));
    pending_hints_.erase(key);
    return CancelledAggregation{std::move(cancelled)};
  }

  /// @brief Cancel all pending aggregation groups whose key matches the
  /// predicate.
  template <typename PredicateFn>
  [[nodiscard]] std::vector<AggregationParticipant> CancelPendingIf(
      PredicateFn predicate /*[in]*/) {
    auto cancelled = shard_aggregator_.CancelIf(predicate);
    std::erase_if(pending_hints_,
                  [&](const auto& e) { return predicate(e.first); });
    return cancelled;
  }

  // Shard aggregation state (pre-dispatch)
  setu::commons::utils::ShardAggregator<CopyKey, CopySpec, CopyKeyHash>
      shard_aggregator_;
  std::unordered_map<CopyKey, PendingHints, CopyKeyHash> pending_hints_;

  // In-flight operation state (post-dispatch)
  std::unordered_map<CopyOperationId, CopyOperationStatePtr,
                     boost::hash<CopyOperationId>>
      copy_operations_;
};

//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
