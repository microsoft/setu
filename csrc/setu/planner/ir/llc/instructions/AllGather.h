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
#include "planner/ir/llc/CommId.h"
#include "planner/ir/llc/ShardAccessTypes.h"
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

/// AllGather collective within the specified communicator.
///
/// Each rank contributes `send_count` elements from its send buffer, and
/// receives the concatenation of all ranks' contributions into its recv
/// buffer.  The recv buffer must be `send_count * num_ranks` elements.
///
/// Maps directly to ncclAllGather in the NCCL backend.
struct AllGather {
  AllGather(CommId comm_id_param, BufferRef send_ref_param,
            std::size_t send_offset_bytes_param, BufferRef recv_ref_param,
            std::size_t recv_offset_bytes_param, std::size_t send_count_param,
            torch::Dtype dtype_param, DeviceRank num_ranks_param,
            DevicePtr send_ptr_param = nullptr,
            DevicePtr recv_ptr_param = nullptr)
      : comm_id(comm_id_param),
        send_ref(std::move(send_ref_param)),
        send_offset_bytes(send_offset_bytes_param),
        recv_ref(std::move(recv_ref_param)),
        recv_offset_bytes(recv_offset_bytes_param),
        send_count(send_count_param),
        dtype(dtype_param),
        num_ranks(num_ranks_param),
        send_ptr(send_ptr_param),
        recv_ptr(recv_ptr_param) {}

  ~AllGather() = default;
  AllGather(const AllGather&) = default;
  AllGather& operator=(const AllGather&) = default;
  AllGather(AllGather&&) = default;
  AllGather& operator=(AllGather&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static AllGather Deserialize(const BinaryRange& range);

  void Embellish(const std::function<DevicePtr(const BufferRef&)>& resolver);

  [[nodiscard]] ShardAccessMap GetShardAccess() const;

  CommId comm_id;
  BufferRef send_ref;
  std::size_t send_offset_bytes;
  BufferRef recv_ref;
  std::size_t recv_offset_bytes;
  std::size_t send_count;
  torch::Dtype dtype;
  DeviceRank num_ranks;

  // Embellished pointers
  DevicePtr send_ptr;
  DevicePtr recv_ptr;
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
