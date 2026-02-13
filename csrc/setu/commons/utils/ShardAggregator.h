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

/// @brief Aggregates shard submissions from multiple sources and fires when all
/// expected shards have arrived.
///
/// Pattern: multi-source aggregation. Each shard submission carries a key that
/// identifies the group (e.g., src/dst tensor pair), a shard_id, a payload for
/// validation, and participant info. When all expected shards have arrived, the
/// completed group is returned and the internal state is cleaned up.
///
/// @tparam KeyType The group key type (must support operator<).
/// @tparam PayloadType The payload type stored per group. Must support
/// operator== for validation of consistency across submissions.
template <typename KeyType, typename PayloadType>
class ShardAggregator {
 public:
  /// @brief Submit a shard for aggregation.
  ///
  /// @param key [in] The group key (e.g., CopyKey{src, dst}).
  /// @param shard_id [in] The shard being submitted.
  /// @param payload [in] The payload for this group. First submission stores
  ///   it; subsequent submissions are validated against the stored payload
  ///   using validate_fn.
  /// @param participant [in] The identity and request_id of the submitter.
  /// @param expected_count [in] Total number of shards expected for this group.
  /// @param validate_fn [in] Callable(const PayloadType& stored, const
  ///   PayloadType& incoming) that asserts payload consistency.
  /// @return CompletedGroup if this submission completes the group, nullopt
  ///   otherwise.
  template <typename ValidateFn>
  [[nodiscard]] std::optional<CompletedGroup<PayloadType>> Submit(
      const KeyType& key /*[in]*/, const ShardId& shard_id /*[in]*/,
      const PayloadType& payload /*[in]*/,
      AggregationParticipant participant /*[in]*/,
      std::size_t expected_count /*[in]*/, ValidateFn validate_fn /*[in]*/) {
    auto& group = groups_[key];

    // Check for duplicate shard submission
    ASSERT_VALID_RUNTIME(!group.shards_received.contains(shard_id),
                         "ShardAggregator: duplicate shard submission: {}",
                         shard_id);

    // Store or validate the payload
    if (!group.payload.has_value()) {
      group.payload.emplace(payload);
    } else {
      validate_fn(group.payload.value(), payload);
    }

    group.shards_received.insert(shard_id);
    group.participants.push_back(std::move(participant));

    // Check if all expected shards have arrived
    if (group.shards_received.size() == expected_count) {
      CompletedGroup<PayloadType> result{std::move(group.payload.value()),
                                         std::move(group.participants)};
      groups_.erase(key);
      return result;
    }

    return std::nullopt;
  }

 private:
  struct PendingGroup {
    std::set<ShardId> shards_received;
    std::optional<PayloadType> payload;
    std::vector<AggregationParticipant> participants;
  };

  std::map<KeyType, PendingGroup> groups_;
};

//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
