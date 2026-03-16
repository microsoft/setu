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
#include "commons/datatypes/TensorSelection.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::telemetry {
//==============================================================================
using setu::commons::BinaryBuffer;
using setu::commons::BinaryRange;
using setu::commons::CopyOperationId;
using setu::commons::datatypes::TensorSelectionPtr;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

/// @brief Timing for a single compiler pass.
struct PassTiming {
  std::string pass_name;
  double elapsed_ms;

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(pass_name, elapsed_ms);
  }

  static PassTiming Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto [name, ms] = reader.ReadFields<std::string, double>();
    return PassTiming{std::move(name), ms};
  }
};

/// @brief Metrics from the Planner::Compile stage (backend-agnostic).
struct CompilationMetrics {
  CopyOperationId copy_op_id;
  double total_compile_time_ms;
  std::vector<PassTiming> pass_timings;
  std::uint32_t num_participants;
  std::vector<std::pair<std::string, std::uint32_t>>
      participant_instruction_counts;

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(copy_op_id, total_compile_time_ms, pass_timings,
                       num_participants);
    // Serialize vector of pairs manually
    writer.Write<std::uint32_t>(
        static_cast<std::uint32_t>(participant_instruction_counts.size()));
    for (const auto& [name, count] : participant_instruction_counts) {
      writer.Write(name);
      writer.Write(count);
    }
  }

  static CompilationMetrics Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto [id, total_ms, timings, num_parts] =
        reader.ReadFields<CopyOperationId, double, std::vector<PassTiming>,
                          std::uint32_t>();

    auto num_entries = reader.Read<std::uint32_t>();
    std::vector<std::pair<std::string, std::uint32_t>> counts;
    counts.reserve(num_entries);
    for (std::uint32_t i = 0; i < num_entries; ++i) {
      auto name = reader.Read<std::string>();
      auto count = reader.Read<std::uint32_t>();
      counts.emplace_back(std::move(name), count);
    }

    return CompilationMetrics{id, total_ms, std::move(timings), num_parts,
                              std::move(counts)};
  }
};

/// @brief End-to-end timing for a complete copy operation.
struct E2EMetrics {
  CopyOperationId copy_op_id;
  double e2e_time_ms;
  std::uint64_t total_bytes_transferred = 0;
  std::string src_name;
  std::string dst_name;
  TensorSelectionPtr src_selection;
  TensorSelectionPtr dst_selection;

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.WriteFields(copy_op_id, e2e_time_ms, total_bytes_transferred,
                       src_name, dst_name, src_selection, dst_selection);
  }

  static E2EMetrics Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    auto [id, ms, bytes, src, dst, src_sel, dst_sel] =
        reader
            .ReadFields<CopyOperationId, double, std::uint64_t, std::string,
                        std::string, TensorSelectionPtr, TensorSelectionPtr>();
    return E2EMetrics{id,
                      ms,
                      bytes,
                      std::move(src),
                      std::move(dst),
                      std::move(src_sel),
                      std::move(dst_sel)};
  }
};

//==============================================================================
}  // namespace setu::telemetry
//==============================================================================
// Forward-declare NCCLWorkerMetrics so the variant can reference it.
namespace setu::telemetry {
struct NCCLWorkerMetrics;
}
//==============================================================================
namespace setu::telemetry {
//==============================================================================

/// @brief Union of all metrics message types.
///
/// Each backend adds its own worker metrics type to this variant.
/// The variant index is serialized as a uint32 prefix.
/// Index 0: NCCLWorkerMetrics (NCCL backend worker metrics)
/// Index 1: CompilationMetrics (backend-agnostic compilation timing)
/// Index 2: E2EMetrics (backend-agnostic end-to-end timing)
using MetricsMessage =
    std::variant<NCCLWorkerMetrics, CompilationMetrics, E2EMetrics>;

//==============================================================================
}  // namespace setu::telemetry
//==============================================================================
