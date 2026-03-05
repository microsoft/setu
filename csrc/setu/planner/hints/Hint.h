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

using CompilerHint = std::variant<RoutingHint, BandwidthHint>;

/// @brief Compute an FNV-1a fingerprint of a hints vector for SPMD
/// consistency verification. Cheap (~nanoseconds for typical hint lists).
[[nodiscard]] inline std::uint64_t Fingerprint(
    const std::vector<CompilerHint>& hints) {
  setu::commons::BinaryBuffer buffer;
  setu::commons::utils::BinaryWriter writer(buffer);
  writer.Write(hints);

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
