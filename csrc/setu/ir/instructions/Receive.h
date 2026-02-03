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
#include "commons/Types.h"
#include "commons/enums/Enums.h"
#include "commons/utils/Serialization.h"
//==============================================================================
#include "setu/ir/ShardRef.h"
//==============================================================================
namespace setu::ir {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

struct Receive {
  Receive(ShardRef dst_shard_param, std::size_t offset_bytes_param,
          std::size_t count_param, torch::Dtype dtype_param,
          DevicePtr dst_ptr_param = nullptr)
      : peer_rank(std::nullopt),
        dst_shard(std::move(dst_shard_param)),
        offset_bytes(offset_bytes_param),
        count(count_param),
        dtype(dtype_param),
        dst_ptr(dst_ptr_param) {}

  ~Receive() = default;
  Receive(const Receive&) = default;
  Receive& operator=(const Receive&) = default;
  Receive(Receive&&) = default;
  Receive& operator=(Receive&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static Receive Deserialize(const BinaryRange& range);

  /**
   * @brief Populates the device pointers by looking up the base address.
   */
  void Embellish(const std::function<DevicePtr(const ShardRef&)>& resolver);

  /**
   * @brief Sets the peer rank for this receive operation.
   *
   * @param rank The rank of the source device in the communicator
   */
  void SetPeerRank(DeviceRank rank) { peer_rank = rank; }

  /**
   * @brief Gets the peer rank, asserting that it has been set.
   *
   * @return The source device rank
   */
  [[nodiscard]] DeviceRank GetPeerRank() const {
    ASSERT_VALID_RUNTIME(peer_rank.has_value(),
                         "Peer rank has not been set for Receive");
    return peer_rank.value();
  }

  std::optional<DeviceRank> peer_rank;
  ShardRef dst_shard;
  std::size_t offset_bytes;
  std::size_t count;
  torch::Dtype dtype;

  // Embellished pointers
  DevicePtr dst_ptr;
};

//==============================================================================
}  // namespace setu::ir
//==============================================================================
