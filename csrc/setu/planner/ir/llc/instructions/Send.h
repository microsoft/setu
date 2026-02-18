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
#include "planner/ir/ref/BufferRef.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::planner::ir::ref::BufferRef;
using setu::planner::ir::ref::ShardRef;
//==============================================================================

/// NCCL point-to-point send to a peer rank within the active communicator.
///
/// Sends `count` elements of type `dtype` starting at `offset_bytes` in
/// `src_ref` to the device identified by `peer_rank`.  The communicator
/// must have been established by a preceding InitComm/UseComm instruction.
struct Send {
  Send(BufferRef src_ref_param, std::size_t offset_bytes_param,
       std::size_t count_param, torch::Dtype dtype_param,
       DeviceRank peer_rank_param, DevicePtr src_ptr_param = nullptr)
      : src_ref(std::move(src_ref_param)),
        offset_bytes(offset_bytes_param),
        count(count_param),
        dtype(dtype_param),
        peer_rank(peer_rank_param),
        src_ptr(src_ptr_param) {}

  ~Send() = default;
  Send(const Send&) = default;
  Send& operator=(const Send&) = default;
  Send(Send&&) = default;
  Send& operator=(Send&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static Send Deserialize(const BinaryRange& range);

  /**
   * @brief Populates the device pointers by looking up the base address.
   */
  void Embellish(const std::function<DevicePtr(const BufferRef&)>& resolver);

  BufferRef src_ref;
  std::size_t offset_bytes;
  std::size_t count;
  torch::Dtype dtype;
  DeviceRank peer_rank;

  // Embellished pointers
  DevicePtr src_ptr;
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
