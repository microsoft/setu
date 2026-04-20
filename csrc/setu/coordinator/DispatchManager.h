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
    std::optional<std::vector<std::string>> pass_names;
  };

  struct CompletedAggregation {
    CopySpec spec;
    std::vector<AggregationParticipant> participants;
    std::vector<setu::planner::hints::CompilerHint> hints;
    std::optional<std::vector<std::string>> pass_names;
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

  /// @brief Submit a shard for aggregation.
  ///
  /// First-writer-wins per group: the first submission's CopySpec and schedule
  /// become authoritative. Subsequent submissions must match both the CopySpec
  /// (via `validate_fn`) and the schedule fingerprint; a mismatch cancels only
  /// that group and returns all participants so the caller can send error
  /// responses.
  template <typename ValidateFn>
  [[nodiscard]] SubmitResult SubmitShard(ShardSubmission submission /*[in]*/,
                                         ValidateFn validate_fn /*[in]*/) {
    CopyKey key{submission.copy_spec.src_name, submission.copy_spec.dst_name};

    const auto schedule_fingerprint = setu::planner::hints::ScheduleFingerprint(
        submission.hints, submission.pass_names);

    GroupPayload group_payload{std::move(submission.copy_spec),
                               std::move(submission.hints),
                               std::move(submission.pass_names),
                               schedule_fingerprint};

    // Save before move — Submit consumes participant, but DispatchManager
    // must re-add the triggering submitter on cancellation (the aggregator
    // returns only previously-accepted participants).
    auto triggering_participant = submission.participant;

    auto outcome = shard_aggregator_.Submit(
        key, submission.shard_id, group_payload,
        std::move(submission.participant), submission.expected_count,
        [&](const GroupPayload& stored, const GroupPayload& incoming) {
          if (stored.schedule_fingerprint != incoming.schedule_fingerprint) {
            LOG_ERROR(
                "SPMD schedule mismatch for {} -> {}: incoming fingerprint={} "
                "stored fingerprint={} — cancelling group",
                incoming.spec.src_name, incoming.spec.dst_name,
                incoming.schedule_fingerprint, stored.schedule_fingerprint);
            return false;
          }
          return validate_fn(stored.spec, incoming.spec);
        });

    return std::visit(
        [&](auto&& alt) -> SubmitResult {
          using T = std::decay_t<decltype(alt)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            return std::monostate{};
          } else if constexpr (std::is_same_v<
                                   T, setu::commons::utils::CompletedGroup<
                                          GroupPayload>>) {
            return CompletedAggregation{std::move(alt.payload.spec),
                                        std::move(alt.participants),
                                        std::move(alt.payload.hints),
                                        std::move(alt.payload.pass_names)};
          } else {
            // CancelledGroup — append triggering submitter so they also
            // receive an error response.
            alt.participants.push_back(std::move(triggering_participant));
            return CancelledAggregation{std::move(alt.participants)};
          }
        },
        std::move(outcome));
  }

  /// @brief Cancel all pending aggregation groups whose src or dst tensor
  /// is in the given set.
  [[nodiscard]] std::vector<AggregationParticipant> CancelPendingByTensors(
      const std::set<TensorName>& tensor_names /*[in]*/) {
    return shard_aggregator_.CancelIf([&tensor_names](const CopyKey& key) {
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
    state->start_time = std::chrono::high_resolution_clock::now();

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
      LOG_DEBUG(
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

  /// @brief CopySpec + schedule stored per aggregation group. Schedule is
  /// per-group so op-K and op-K+1 for the same (src, dst) can carry
  /// different schedules.
  struct GroupPayload {
    CopySpec spec;
    std::vector<setu::planner::hints::CompilerHint> hints;
    std::optional<std::vector<std::string>> pass_names;
    std::uint64_t schedule_fingerprint;
  };

  // Shard aggregation state (pre-dispatch)
  setu::commons::utils::ShardAggregator<CopyKey, GroupPayload, CopyKeyHash>
      shard_aggregator_;

  // In-flight operation state (post-dispatch)
  std::unordered_map<CopyOperationId, CopyOperationStatePtr,
                     boost::hash<CopyOperationId>>
      copy_operations_;
};

//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
