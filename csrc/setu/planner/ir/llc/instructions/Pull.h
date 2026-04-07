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
#include "planner/ir/llc/ShardAccessTypes.h"
#include "planner/ir/ref/BufferRef.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::DevicePtr;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
using setu::planner::ir::ref::BufferRef;
using setu::planner::ir::ref::ShardRef;
//==============================================================================

/// P2P pull (receiver-side DMA) from a remote device on the same node.
///
/// The executing worker calls cudaMemcpyPeerAsync to pull `count` elements
/// of type `dtype` from `src_ref` on `src_device` into local `dst_ref`.
/// Follows UCX/RDMA GET semantics.
struct Pull {
  Pull(BufferRef src_ref_param, std::size_t src_offset_bytes_param,
       BufferRef dst_ref_param, std::size_t dst_offset_bytes_param,
       std::size_t count_param, torch::Dtype dtype_param,
       std::int32_t src_device_param,
       DevicePtr src_ptr_param = nullptr, DevicePtr dst_ptr_param = nullptr)
      : src_ref(std::move(src_ref_param)),
        src_offset_bytes(src_offset_bytes_param),
        dst_ref(std::move(dst_ref_param)),
        dst_offset_bytes(dst_offset_bytes_param),
        count(count_param),
        dtype(dtype_param),
        src_device(src_device_param),
        src_ptr(src_ptr_param),
        dst_ptr(dst_ptr_param) {}

  ~Pull() = default;
  Pull(const Pull&) = default;
  Pull& operator=(const Pull&) = default;
  Pull(Pull&&) = default;
  Pull& operator=(Pull&&) = default;

  [[nodiscard]] std::string ToString() const;

  void Serialize(BinaryBuffer& buffer) const;

  static Pull Deserialize(const BinaryRange& range);

  void Embellish(const std::function<DevicePtr(const BufferRef&)>& resolver);

  [[nodiscard]] ShardAccessMap GetShardAccess() const;

  BufferRef src_ref;
  std::size_t src_offset_bytes;
  BufferRef dst_ref;
  std::size_t dst_offset_bytes;
  std::size_t count;
  torch::Dtype dtype;
  std::int32_t src_device;

  // Embellished pointers
  DevicePtr src_ptr;
  DevicePtr dst_ptr;
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
