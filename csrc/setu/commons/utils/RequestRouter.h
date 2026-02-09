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

/// @brief Tracks which client sent a request so the response can be routed back.
///
/// Pattern: forward-and-route-back. When a NodeAgent forwards a client request
/// to the Coordinator, it calls TrackRequest to remember the originating client
/// identity. When the Coordinator responds, ClaimIdentity retrieves (and
/// removes) the identity so the response can be routed to the correct client.
class RequestRouter {
 public:
  /// @brief Record the client identity that sent a request.
  /// @param request_id [in] The unique request identifier.
  /// @param identity [in] The client identity to associate with this request.
  void TrackRequest(const RequestId& request_id /*[in]*/,
                    const Identity& identity /*[in]*/) {
    auto [_, inserted] = map_.emplace(request_id, identity);
    ASSERT_VALID_RUNTIME(inserted,
                         "RequestRouter: duplicate request_id: {}",
                         request_id);
  }

  /// @brief Retrieve and remove the client identity for a request.
  /// @param request_id [in] The request identifier to look up.
  /// @return The client identity if found, std::nullopt otherwise.
  [[nodiscard]] std::optional<Identity> ClaimIdentity(
      const RequestId& request_id /*[in]*/) {
    auto it = map_.find(request_id);
    if (it == map_.end()) {
      return std::nullopt;
    }
    Identity identity = std::move(it->second);
    map_.erase(it);
    return identity;
  }

 private:
  std::unordered_map<RequestId, Identity> map_;
};

//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
