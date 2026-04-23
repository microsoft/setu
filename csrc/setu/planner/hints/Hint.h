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
#include "commons/utils/Serialization.h"
//==============================================================================
#include "planner/Participant.h"
#include "planner/topo/Topology.h"
//==============================================================================
namespace setu::planner::hints {
//==============================================================================
using setu::planner::Participant;
using setu::planner::topo::Path;
//==============================================================================

struct RoutingHint {
  Participant src;
  Participant dst;
  Path path;

  RoutingHint() = default;

  RoutingHint(Participant src_param, Participant dst_param, Path path_param)
      : src(std::move(src_param)),
        dst(std::move(dst_param)),
        path(std::move(path_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format("RoutingHint(src={}, dst={}, path={})", src, dst, path);
  }

  void Serialize(setu::commons::BinaryBuffer& buffer) const {
    setu::commons::utils::BinaryWriter writer(buffer);
    writer.WriteFields(src, dst, path);
  }

  static RoutingHint Deserialize(const setu::commons::BinaryRange& range) {
    setu::commons::utils::BinaryReader reader(range);
    auto [s, d, p] = reader.ReadFields<Participant, Participant, Path>();
    return RoutingHint(std::move(s), std::move(d), std::move(p));
  }
};

struct BandwidthHint {
  Participant src;
  Participant dst;
  std::vector<Path> paths;
  std::vector<float> weights;  // fractional, must sum to ~1.0

  BandwidthHint() = default;

  BandwidthHint(Participant src_param, Participant dst_param,
                std::vector<Path> paths_param, std::vector<float> weights_param)
      : src(std::move(src_param)),
        dst(std::move(dst_param)),
        paths(std::move(paths_param)),
        weights(std::move(weights_param)) {
    ASSERT_VALID_ARGUMENTS(
        paths.size() == weights.size(),
        "BandwidthHint: paths.size()={} != weights.size()={}", paths.size(),
        weights.size());
    ASSERT_VALID_ARGUMENTS(!paths.empty(),
                           "BandwidthHint: must have at least one path");
  }

  [[nodiscard]] std::string ToString() const {
    return std::format("BandwidthHint(src={}, dst={}, num_paths={})", src, dst,
                       paths.size());
  }

  void Serialize(setu::commons::BinaryBuffer& buffer) const {
    setu::commons::utils::BinaryWriter writer(buffer);
    writer.WriteFields(src, dst, paths, weights);
  }

  static BandwidthHint Deserialize(const setu::commons::BinaryRange& range) {
    setu::commons::utils::BinaryReader reader(range);
    auto [s, d, p, w] =
        reader.ReadFields<Participant, Participant, std::vector<Path>,
                          std::vector<float>>();
    return BandwidthHint(std::move(s), std::move(d), std::move(p),
                         std::move(w));
  }
};

struct PipelineChunkSizeHint {
  std::size_t chunk_size_bytes;

  PipelineChunkSizeHint() = default;

  explicit PipelineChunkSizeHint(std::size_t chunk_size_bytes_param)
      : chunk_size_bytes(chunk_size_bytes_param) {
    ASSERT_VALID_ARGUMENTS(chunk_size_bytes > 0,
                           "PipelineChunkSizeHint: chunk_size_bytes must be "
                           "positive, got {}",
                           chunk_size_bytes);
  }

  [[nodiscard]] std::string ToString() const {
    return std::format("PipelineChunkSizeHint(chunk_size_bytes={})",
                       chunk_size_bytes);
  }

  void Serialize(setu::commons::BinaryBuffer& buffer) const {
    setu::commons::utils::BinaryWriter writer(buffer);
    writer.WriteFields(chunk_size_bytes);
  }

  static PipelineChunkSizeHint Deserialize(
      const setu::commons::BinaryRange& range) {
    setu::commons::utils::BinaryReader reader(range);
    auto [sz] = reader.ReadFields<std::size_t>();
    return PipelineChunkSizeHint(sz);
  }
};

enum class ReplicationStrategy { kAllGather, kNaive, kBatchedCopy };

struct ReplicationHint {
  setu::commons::TensorName dst_name;
  ReplicationStrategy strategy;

  ReplicationHint() = default;

  ReplicationHint(setu::commons::TensorName dst_name_param,
                  ReplicationStrategy strategy_param)
      : dst_name(std::move(dst_name_param)), strategy(strategy_param) {}

  [[nodiscard]] std::string ToString() const {
    const char* strategy_str = "Unknown";
    switch (strategy) {
      case ReplicationStrategy::kAllGather:
        strategy_str = "AllGather";
        break;
      case ReplicationStrategy::kNaive:
        strategy_str = "Naive";
        break;
      case ReplicationStrategy::kBatchedCopy:
        strategy_str = "BatchedCopy";
        break;
    }
    return std::format("ReplicationHint(dst_name={}, strategy={})", dst_name,
                       strategy_str);
  }

  void Serialize(setu::commons::BinaryBuffer& buffer) const {
    setu::commons::utils::BinaryWriter writer(buffer);
    writer.WriteFields(dst_name, static_cast<std::int32_t>(strategy));
  }

  static ReplicationHint Deserialize(const setu::commons::BinaryRange& range) {
    setu::commons::utils::BinaryReader reader(range);
    auto [name, strat_int] =
        reader.ReadFields<setu::commons::TensorName, std::int32_t>();
    return ReplicationHint(std::move(name),
                           static_cast<ReplicationStrategy>(strat_int));
  }
};

using CompilerHint = std::variant<RoutingHint, BandwidthHint, ReplicationHint,
                                  PipelineChunkSizeHint>;

/// @brief Compute an FNV-1a fingerprint over a schedule for SPMD consistency
/// verification.
[[nodiscard]] inline std::uint64_t ScheduleFingerprint(
    const std::vector<CompilerHint>& hints /*[in]*/,
    const std::optional<std::vector<std::string>>& pass_names /*[in]*/) {
  setu::commons::BinaryBuffer buffer;
  setu::commons::utils::BinaryWriter writer(buffer);
  writer.Write(hints);
  writer.Write(pass_names);

  // FNV-1a over the serialized bytes
  constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  std::uint64_t hash = kFnvOffset;
  for (const auto& byte : buffer) {
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= kFnvPrime;
  }
  return hash;
}

//==============================================================================
}  // namespace setu::planner::hints
//==============================================================================
