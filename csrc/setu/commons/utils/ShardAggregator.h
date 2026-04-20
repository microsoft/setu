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
#include "commons/StdCommon.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/Types.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================

/// @brief Participant in a shard aggregation group.
struct AggregationParticipant {
  Identity identity;
  RequestId request_id;
};

/// @brief Result returned when all expected shards have been submitted.
///
/// @tparam PayloadType The payload type stored per group (e.g., CopySpec).
template <typename PayloadType>
struct CompletedGroup {
  PayloadType payload;
  std::vector<AggregationParticipant> participants;
};

/// @brief Result returned when a submission fails validation and the target
/// group is cancelled.
///
/// Contract: `participants` contains only submitters whose shards were already
/// accepted into the group. The submitter whose submission triggered the
/// validation failure is NOT included — callers must append it themselves if
/// an error response to that submitter is required.
struct CancelledGroup {
  std::vector<AggregationParticipant> participants;
};

/// @brief Aggregates shard submissions from multiple sources and fires when all
/// expected shards have arrived.
///
/// Pattern: multi-source aggregation. Each shard submission carries a key that
/// identifies the group (e.g., src/dst tensor pair), a shard_id, a payload for
/// validation, and participant info. When all expected shards have arrived, the
/// completed group is returned and its state is cleaned up.
///
/// Multiple groups per key: the aggregator maintains a FIFO list of open
/// groups per key. A new submission lands in the first group (head→tail) whose
/// `shards_received` does not yet contain this shard_id; if the shard_id is
/// already present in every open group, a fresh group is opened at the tail.
/// This models the invariant "exactly the set of shard owners submits, exactly
/// once per op": a duplicate shard_id under the same key is provably the next
/// op for that owner, not a protocol bug. Per-sender message ordering
/// (preserved by ZMQ DEALER→ROUTER) ensures a given owner's op-K always lands
/// in the K-th group containing that owner.
///
/// Completion order is NOT guaranteed to be FIFO — a group in the middle can
/// complete before the head if its last peer arrives first. List position
/// only determines where a new arrival lands.
///
/// @tparam KeyType The group key type (must be hashable via KeyHash).
/// @tparam PayloadType The payload type stored per group.
/// @tparam KeyHash Hash function object for KeyType.
/// @tparam KeyEqual Equality function object for KeyType.
template <typename KeyType, typename PayloadType,
          typename KeyHash = boost::hash<KeyType>,
          typename KeyEqual = std::equal_to<KeyType>>
class ShardAggregator {
 public:
  using SubmitOutcome =
      std::variant<std::monostate, CompletedGroup<PayloadType>, CancelledGroup>;

  /// @brief Submit a shard for aggregation.
  ///
  /// @param key [in] The group key (e.g., CopyKey{src, dst}).
  /// @param shard_id [in] The shard being submitted.
  /// @param payload [in] The payload for this group. First submission into a
  ///   group stores it; subsequent submissions are validated against the
  ///   stored payload using `validate_fn`.
  /// @param participant [in] The identity and request_id of the submitter.
  /// @param expected_count [in] Total number of shards expected for this
  ///   group. Must be stable for all submissions targeting the same open
  ///   group — an inconsistency fires an assertion.
  /// @param validate_fn [in] Callable(const PayloadType& stored, const
  ///   PayloadType& incoming) → bool. Returns true if payloads are consistent,
  ///   false to reject. Rejection pops the targeted group only.
  /// @return `std::monostate` if the group is still waiting for more shards;
  ///   `CompletedGroup` if this submission completed the target group;
  ///   `CancelledGroup` if validation rejected the submission (the returned
  ///   participants are the previously-accepted ones; the triggering
  ///   submitter is NOT included).
  template <typename ValidateFn>
  [[nodiscard]] SubmitOutcome Submit(
      const KeyType& key /*[in]*/, const ShardId& shard_id /*[in]*/,
      const PayloadType& payload /*[in]*/,
      AggregationParticipant participant /*[in]*/,
      std::size_t expected_count /*[in]*/, ValidateFn validate_fn /*[in]*/) {
    auto& open = groups_[key];

    // Find the first group (head→tail) whose shards_received does not yet
    // contain this shard_id. If every open group already has it, open a new
    // group at the tail — this submission is the next op for that owner.
    auto target = open.end();
    for (auto it = open.begin(); it != open.end(); ++it) {
      if (!it->shards_received.contains(shard_id)) {
        target = it;
        break;
      }
    }
    if (target == open.end()) {
      open.emplace_back();
      target = std::prev(open.end());
    }

    auto& group = *target;

    if (!group.payload.has_value()) {
      // Fresh group: stamp payload and expected_count.
      group.payload.emplace(payload);
      group.expected_count = expected_count;
    } else {
      // Existing group: expected_count must be stable.
      ASSERT_VALID_RUNTIME(
          group.expected_count == expected_count,
          "ShardAggregator: expected_count mismatch for open group "
          "(stored={}, incoming={})",
          group.expected_count, expected_count);

      if (!validate_fn(group.payload.value(), payload)) {
        CancelledGroup result{std::move(group.participants)};
        open.erase(target);
        if (open.empty()) {
          groups_.erase(key);
        }
        return result;
      }
    }

    group.shards_received.insert(shard_id);
    group.participants.push_back(std::move(participant));

    if (group.shards_received.size() == expected_count) {
      CompletedGroup<PayloadType> result{std::move(group.payload.value()),
                                         std::move(group.participants)};
      open.erase(target);
      if (open.empty()) {
        groups_.erase(key);
      }
      return result;
    }

    return std::monostate{};
  }

  /// @brief Cancel and remove every open group for the given key.
  ///
  /// @param key [in] The group key to cancel.
  /// @return All participants from every cancelled group for this key.
  [[nodiscard]] std::vector<AggregationParticipant> Cancel(
      const KeyType& key /*[in]*/) {
    std::vector<AggregationParticipant> cancelled_participants;
    auto it = groups_.find(key);
    if (it != groups_.end()) {
      for (auto& group : it->second) {
        for (auto& p : group.participants) {
          cancelled_participants.push_back(std::move(p));
        }
      }
      groups_.erase(it);
    }
    return cancelled_participants;
  }

  /// @brief Cancel and remove every open group whose key matches the
  /// predicate.
  ///
  /// Used to clean up partially-aggregated groups when the shards involved
  /// are being deregistered (e.g., client disconnect).
  ///
  /// @param predicate [in] Callable(const KeyType&) returning true for keys
  ///   to cancel.
  /// @return All participants from cancelled groups, flattened.
  template <typename PredicateFn>
  [[nodiscard]] std::vector<AggregationParticipant> CancelIf(
      PredicateFn predicate /*[in]*/) {
    std::vector<AggregationParticipant> cancelled_participants;
    auto it = groups_.begin();
    while (it != groups_.end()) {
      if (predicate(it->first)) {
        for (auto& group : it->second) {
          for (auto& p : group.participants) {
            cancelled_participants.push_back(std::move(p));
          }
        }
        it = groups_.erase(it);
      } else {
        ++it;
      }
    }
    return cancelled_participants;
  }

 private:
  struct PendingGroup {
    std::set<ShardId> shards_received;
    std::optional<PayloadType> payload;
    std::vector<AggregationParticipant> participants;
    std::size_t expected_count{0};
  };

  std::unordered_map<KeyType, std::list<PendingGroup>, KeyHash, KeyEqual>
      groups_;
};

//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
