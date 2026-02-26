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

/// @brief Dependency tracker: waiters are released when all their blockers
/// resolve.
///
/// ## Components
///
/// - **Blocker**: something that must finish (e.g., a copy operation ID).
///   Resolved by calling Resolve(blocker_id).
/// - **Waiter**: something waiting for one or more blockers to finish
///   (e.g., a client identity). Added via AddWaiter(waiter_id, {blockers}).
///   Released (returned from Resolve) when its last remaining blocker
///   resolves.
/// - **Payload** (optional): data associated with a waiter. When provided,
///   Resolve returns payloads instead of waiter IDs.
///
/// ## Expressivity
///
/// The relationship is fully M:N. Any number of waiters can depend on any
/// subset of blockers, and blockers can be shared across waiters. A waiter
/// is released only when ALL of its blockers have been resolved.
///
/// Resolve is one-shot: calling Resolve(X) twice returns results only on
/// the first call. Dependencies are declared at AddWaiter time and cannot
/// be modified afterward.
///
/// ## Lost wakeup handling
///
/// If AddWaiter is always called before Resolve, no special handling is
/// needed — the blocker is tracked implicitly via the reverse index.
///
/// A lost wakeup occurs when a blocker finishes before the waiter
/// registers. Resolve already fired and won't be called again, so the
/// waiter would be stuck forever.
///
/// RegisterBlocker prevents this. It tells PendingOperations that a
/// blocker exists and might finish early. When Resolve is called on a
/// registered blocker, a tombstone is recorded — a marker that says "this
/// blocker already finished." When a waiter later calls AddWaiter and all
/// its blockers have tombstones, the result is returned immediately.
///
/// RegisterBlocker is refcounted: calling it N times for the same key
/// allows N late AddWaiter calls to be served. Each one decrements the
/// refcount, and the tombstone is cleaned up when it hits zero.
///
/// RemoveBlocker force-clears a blocker's registration, tombstone, and
/// reverse index. Use it when the blocker is being torn down and no more
/// waiters will arrive for it.
///
/// **Memory contract**: every registered blocker must eventually be
/// consumed by a late AddWaiter or explicitly cleaned up via
/// RemoveBlocker. Otherwise the tombstone leaks.
///
/// ## Examples
///
/// ### Without payload — Resolve returns waiter IDs
/// @code
///   PendingOperations<std::string, std::int32_t> pending;
///
///   pending.AddWaiter("client_1", {42});
///   pending.AddWaiter("client_2", {42});
///
///   auto released = pending.Resolve(42);
///   // released == {"client_1", "client_2"}
/// @endcode
///
/// ### With payload — Resolve returns payloads
/// @code
///   PendingOperations<std::string, std::string, MyRequest> pending;
///
///   pending.AddWaiter("req_1", {"copy_A", "copy_B"}, MyRequest{...});
///
///   pending.Resolve("copy_A");  // returns {} — still waiting on copy_B
///   auto r = pending.Resolve("copy_B");
///   // r == {MyRequest{...}} — all blockers done, payload returned
/// @endcode
///
/// ### Lost wakeup handling
/// @code
///   PendingOperations<std::string, std::int32_t> pending;
///
///   pending.RegisterBlocker(7);
///   pending.Resolve(7);  // no waiters yet — tombstone is kept
///
///   auto r = pending.AddWaiter("late_client", {7});
///   // r == "late_client" — returned immediately, tombstone consumed
/// @endcode
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

  /// @brief Register a blocker for lost wakeup handling.
  ///
  /// Only needed when Resolve may be called before AddWaiter. If the
  /// caller guarantees AddWaiter always precedes Resolve, registration
  /// is unnecessary.
  ///
  /// When a registered blocker is resolved, a tombstone is kept so that
  /// a subsequent AddWaiter returns immediately instead of waiting
  /// forever.
  ///
  /// Refcounted: calling RegisterBlocker(X) N times allows N late
  /// AddWaiter calls to be served before the tombstone is cleaned up.
  ///
  /// Every registration must eventually be consumed by a late AddWaiter
  /// or cleaned up via RemoveBlocker. Otherwise the tombstone leaks.
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
