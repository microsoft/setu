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
//==============================================================================
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::telemetry {
//==============================================================================
using setu::commons::BinaryBuffer;
using setu::commons::BinaryRange;
using setu::commons::CopyOperationId;
using setu::commons::DeviceRank;
using setu::commons::NodeId;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

/// @brief Timing for a single NCCL group (between barriers).
struct NCCLGroupTiming {
  std::uint32_t group_index;
  double elapsed_ms;    // GPU-measured via CUDA events
  std::size_t num_ops;  // number of Copy/Send/Receive in this group

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(group_index, elapsed_ms,
                       static_cast<std::uint64_t>(num_ops));
  }

  static NCCLGroupTiming Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto [idx, ms, ops] =
        reader.ReadFields<std::uint32_t, double, std::uint64_t>();
    return NCCLGroupTiming{idx, ms, static_cast<std::size_t>(ops)};
  }
};

/// @brief NCCL-specific worker metrics: per-group GPU timing.
struct NCCLWorkerMetrics {
  CopyOperationId copy_op_id;
  NodeId node_id;
  DeviceRank device_rank;
  std::vector<NCCLGroupTiming> group_timings;
  double total_execute_ms;  // wall-clock for entire Execute()

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(copy_op_id, node_id, device_rank, group_timings,
                       total_execute_ms);
  }

  static NCCLWorkerMetrics Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto [id, nid, rank, timings, total] =
        reader.ReadFields<CopyOperationId, NodeId, DeviceRank,
                          std::vector<NCCLGroupTiming>, double>();
    return NCCLWorkerMetrics{id, nid, rank, std::move(timings), total};
  }
};

//==============================================================================
}  // namespace setu::telemetry
//==============================================================================
