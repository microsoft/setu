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
#include "commons/Types.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================

/// @brief Manages clients waiting for an async operation keyed by KeyType.
///
/// Pattern: wait-and-notify. Clients register as waiters for a given key via
/// AddWaiter. When the operation completes, DrainWaiters retrieves all waiting
/// identities and removes the entry in one atomic step.
///
/// @tparam KeyType The key type (e.g., CopyOperationId, ShardId).
/// @tparam Hash The hash function for KeyType (defaults to boost::hash).
template <typename KeyType, typename Hash = boost::hash<KeyType>>
class PendingWaits {
 public:
  /// @brief Register a client as waiting for the given key.
  /// @param key [in] The operation key to wait on.
  /// @param identity [in] The client identity to register.
  void AddWaiter(const KeyType& key /*[in]*/,
                 const Identity& identity /*[in]*/) {
    map_[key].push_back(identity);
  }

  /// @brief Retrieve and remove all waiters for the given key.
  /// @param key [in] The operation key whose waiters to drain.
  /// @return Vector of waiting client identities. Empty if no waiters.
  [[nodiscard]] std::vector<Identity> DrainWaiters(
      const KeyType& key /*[in]*/) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return {};
    }
    std::vector<Identity> waiters = std::move(it->second);
    map_.erase(it);
    return waiters;
  }

  /// @brief Check whether any waiters exist for the given key.
  /// @param key [in] The operation key to check.
  /// @return True if at least one waiter is registered.
  [[nodiscard]] bool HasWaiters(const KeyType& key /*[in]*/) const {
    return map_.contains(key);
  }

 private:
  std::unordered_map<KeyType, std::vector<Identity>, Hash> map_;
};

//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
