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
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================

/// @brief Bipartite waiter-blocker graph for tracking pending operations.
///
/// Models a many-to-many relationship between waiters and blockers:
///   - **N:1** (many waiters on one blocker): RegisterBlocker, AddWaiter
///     (multiple), Resolve → returns all unblocked waiters.
///   - **1:N** (one waiter blocked by many keys): AddWaiter with multiple
///     blockers, Resolve each → waiter returned when last blocker resolves.
///
/// Late-arrival support: RegisterBlocker + Resolve before AddWaiter causes
/// AddWaiter to return immediately (consuming one registration refcount).
///
/// @tparam WaiterId      Key type identifying each waiter.
/// @tparam BlockerId     Key type identifying each blocker.
/// @tparam WaiterPayload Optional payload type stored with each waiter.
///                        When void, the WaiterId itself serves as the result.
/// @tparam WaiterHash    Hash function for WaiterId.
/// @tparam BlockerHash   Hash function for BlockerId.
template <typename WaiterId, typename BlockerId, typename WaiterPayload = void,
          typename WaiterHash = boost::hash<WaiterId>,
          typename BlockerHash = boost::hash<BlockerId>>
class PendingOperations {
  static constexpr bool kHasPayload = !std::is_void_v<WaiterPayload>;
  using StoredPayload =
      std::conditional_t<kHasPayload, WaiterPayload, std::monostate>;
  using ResultType = std::conditional_t<kHasPayload, WaiterPayload, WaiterId>;

  struct WaiterEntry {
    std::set<BlockerId> blockers;
    [[no_unique_address]] StoredPayload payload{};
  };

 public:
  // --- Blocker lifecycle ---

  /// @brief Register a blocker key for late-arrival support.
  ///
  /// Enables late-arrival handling: if Resolve() is called before AddWaiter(),
  /// subsequent AddWaiter() calls whose blockers are all already-resolved
  /// return the result immediately. Refcounted: multiple calls for the same
  /// key are allowed.
  ///
  /// @param key [in] The blocker key to register.
  void RegisterBlocker(const BlockerId& key /*[in]*/) { registered_[key]++; }

  /// @brief Check whether a blocker key is registered (refcount > 0).
  ///
  /// @param key [in] The blocker key to check.
  /// @return True if the key has been registered and refcount > 0.
  [[nodiscard]] bool IsBlockerRegistered(const BlockerId& key /*[in]*/) const {
    return registered_.contains(key);
  }

  /// @brief Remove a blocker: erases registration, resolved tombstone, and
  /// reverse-index entry.
  ///
  /// Does NOT release waiters — use Resolve for that.
  ///
  /// @param key [in] The blocker key to remove.
  void RemoveBlocker(const BlockerId& key /*[in]*/) {
    registered_.erase(key);
    resolved_.erase(key);
    blocker_to_waiters_.erase(key);
  }

  // --- Waiter lifecycle ---

  /// @brief Add a waiter blocked by a set of blocker keys (void payload).
  ///
  /// If all blockers are already resolved (tombstoned), returns the WaiterId
  /// immediately (late arrival) and decrements registration refcounts.
  /// Otherwise, stores the waiter internally.
  ///
  /// @param waiter_id [in] Unique identifier for this waiter.
  /// @param blockers  [in] The set of blocker keys this waiter depends on.
  /// @return The WaiterId if all blockers are already resolved; std::nullopt
  ///         if the waiter was stored.
  [[nodiscard]] std::optional<ResultType> AddWaiter(
      WaiterId waiter_id /*[in]*/, std::set<BlockerId> blockers /*[in]*/)
    requires(!kHasPayload)
  {
    return AddWaiterImpl(std::move(waiter_id), std::move(blockers),
                         std::monostate{});
  }

  /// @brief Add a waiter blocked by a set of blocker keys (with payload).
  ///
  /// If all blockers are already resolved (tombstoned), returns the payload
  /// immediately (late arrival) and decrements registration refcounts.
  /// Otherwise, stores the waiter internally.
  ///
  /// @param waiter_id [in] Unique identifier for this waiter.
  /// @param blockers  [in] The set of blocker keys this waiter depends on.
  /// @param payload   [in] The data to associate with this waiter.
  /// @return The payload if all blockers are already resolved; std::nullopt
  ///         if the waiter was stored.
  template <typename P = StoredPayload>
    requires(kHasPayload)
  [[nodiscard]] std::optional<ResultType> AddWaiter(
      WaiterId waiter_id /*[in]*/, std::set<BlockerId> blockers /*[in]*/,
      P payload /*[in]*/) {
    return AddWaiterImpl(std::move(waiter_id), std::move(blockers),
                         StoredPayload{std::move(payload)});
  }

  /// @brief Remove a waiter and clean up its blocker references.
  ///
  /// @param waiter_id [in] The waiter to remove.
  void RemoveWaiter(const WaiterId& waiter_id /*[in]*/) {
    auto it = waiters_.find(waiter_id);
    if (it == waiters_.end()) {
      return;
    }

    for (const auto& key : it->second.blockers) {
      auto rev_it = blocker_to_waiters_.find(key);
      if (rev_it != blocker_to_waiters_.end()) {
        rev_it->second.erase(waiter_id);
        if (rev_it->second.empty()) {
          blocker_to_waiters_.erase(rev_it);
        }
      }
    }

    waiters_.erase(it);
  }

  // --- Resolution ---

  /// @brief A blocker has been resolved.
  ///
  /// Removes it from all waiters' blocker sets. Returns results of waiters
  /// now fully unblocked. Also records as tombstone if the blocker was
  /// registered (for late-arrival support).
  ///
  /// @param key [in] The blocker key that has been resolved.
  /// @return Vector of WaiterIds (void payload) or WaiterPayloads (non-void)
  ///         for waiters now fully unblocked.
  [[nodiscard]] std::vector<ResultType> Resolve(const BlockerId& key /*[in]*/) {
    // Record tombstone only if this blocker was registered (for late-arrival)
    if (registered_.contains(key)) {
      resolved_.insert(key);
    }

    std::vector<ResultType> unblocked;

    auto idx_it = blocker_to_waiters_.find(key);
    if (idx_it == blocker_to_waiters_.end()) {
      return unblocked;
    }

    // Move out the set of affected waiter IDs and erase the index entry
    auto waiter_ids = std::move(idx_it->second);
    blocker_to_waiters_.erase(idx_it);

    for (const auto& waiter_id : waiter_ids) {
      auto w_it = waiters_.find(waiter_id);
      if (w_it == waiters_.end()) {
        continue;
      }

      w_it->second.blockers.erase(key);

      if (w_it->second.blockers.empty()) {
        if constexpr (kHasPayload) {
          unblocked.push_back(std::move(w_it->second.payload));
        } else {
          unblocked.push_back(waiter_id);
        }
        waiters_.erase(w_it);
      }
    }

    return unblocked;
  }

 private:
  [[nodiscard]] std::optional<ResultType> AddWaiterImpl(
      WaiterId waiter_id /*[in]*/, std::set<BlockerId> blockers /*[in]*/,
      StoredPayload payload /*[in]*/) {
    ASSERT_VALID_ARGUMENTS(!blockers.empty(),
                           "PendingOperations::AddWaiter requires at least one "
                           "blocker key");

    // Check if all blockers are already resolved (tombstoned)
    bool all_resolved = true;
    for (const auto& key : blockers) {
      if (!resolved_.contains(key)) {
        all_resolved = false;
        break;
      }
    }

    if (all_resolved) {
      // Late arrival: all blockers already resolved — return immediately
      for (const auto& key : blockers) {
        DecrementRegistration(key);
      }
      if constexpr (kHasPayload) {
        return std::move(payload);
      } else {
        return std::move(waiter_id);
      }
    }

    // Build reverse index for unresolved blockers
    for (const auto& key : blockers) {
      if (!resolved_.contains(key)) {
        blocker_to_waiters_[key].insert(waiter_id);
      }
    }

    // Remove already-resolved blockers from the waiter's set
    std::erase_if(blockers, [this](const BlockerId& key) {
      return resolved_.contains(key);
    });

    waiters_.emplace(std::move(waiter_id),
                     WaiterEntry{std::move(blockers), std::move(payload)});
    return std::nullopt;
  }

  /// @brief Decrement registration refcount for a blocker key.
  ///
  /// Called during late-arrival AddWaiter. When refcount hits 0, erases
  /// both the registration and the resolved tombstone.
  void DecrementRegistration(const BlockerId& key /*[in]*/) {
    auto it = registered_.find(key);
    if (it == registered_.end()) {
      return;
    }
    if (it->second <= 1) {
      registered_.erase(it);
      resolved_.erase(key);
    } else {
      it->second--;
    }
  }

  /// Forward map: waiter ID → waiter state (blocker set + optional payload)
  std::unordered_map<WaiterId, WaiterEntry, WaiterHash> waiters_;

  /// Reverse index: blocker key → set of waiter IDs blocked by that key
  std::unordered_map<BlockerId, std::set<WaiterId>, BlockerHash>
      blocker_to_waiters_;

  /// Tombstones: blocker keys that have been resolved (for late-arrival).
  /// Only populated for registered blockers.
  std::unordered_set<BlockerId, BlockerHash> resolved_;

  /// Registration refcounts (controls late-arrival tombstone lifetime)
  std::unordered_map<BlockerId, std::size_t, BlockerHash> registered_;
};

//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
