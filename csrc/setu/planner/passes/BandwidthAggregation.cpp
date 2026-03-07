#include "planner/passes/BandwidthAggregation.h"

#include "commons/Logging.h"
#include "planner/hints/Hint.h"

namespace setu::planner::passes {

using setu::planner::Participant;
using setu::planner::hints::BandwidthHint;
using setu::planner::hints::RoutingHint;
using setu::planner::topo::Link;
using setu::planner::topo::Path;

//==============================================================================
// Helpers
//==============================================================================

/// Compute per-path element counts proportional to bottleneck bandwidth.
///
/// Given N paths with bottleneck bandwidths [bw_0, ..., bw_{N-1}] and a total
/// of E elements, path i receives floor(E * bw_i / sum(bw)) elements.
/// Rounding remainder is assigned to the highest-bandwidth path to minimize
/// overall transfer time.
///
/// Example: 3 paths with bandwidths [200, 100, 100] GB/s and 400k elements
///          → splits = [200k, 100k, 100k].
static std::vector<std::size_t> ComputeSplits(const std::vector<Path>& paths,
                                              std::size_t total_elements) {
  ASSERT_VALID_ARGUMENTS(!paths.empty(), "ComputeSplits requires >= 1 path");

  float total_bw = 0.0f;
  for (const auto& p : paths) {
    total_bw += p.bottleneck_bandwidth_gbps;
  }
  ASSERT_VALID_RUNTIME(total_bw > 0.0f,
                       "Total bandwidth across paths must be positive");

  // Compute floor(E * bw_i / total_bw) per path, tracking the total assigned
  // and the highest-bandwidth path to absorb the rounding remainder.
  std::vector<std::size_t> splits(paths.size());
  std::size_t assigned = 0;
  std::size_t best_idx = 0;
  float best_bw = 0.0f;

  for (std::size_t i = 0; i < paths.size(); ++i) {
    float fraction = paths[i].bottleneck_bandwidth_gbps / total_bw;
    splits[i] = static_cast<std::size_t>(static_cast<double>(fraction) *
                                         static_cast<double>(total_elements));
    assigned += splits[i];
    if (paths[i].bottleneck_bandwidth_gbps > best_bw) {
      best_bw = paths[i].bottleneck_bandwidth_gbps;
      best_idx = i;
    }
  }

  splits[best_idx] += total_elements - assigned;

  return splits;
}

/// Estimate the wall-clock transfer time when paths run in parallel.
///
/// Each path transfers its chunk independently, so the overall time is the
/// maximum across all paths:
///   time = max_i(path_i.total_latency_us + chunk_bytes_i / (bw_i * 1e3))
static float EstimateTransferTime(const std::vector<Path>& paths,
                                  const std::vector<std::size_t>& splits,
                                  std::size_t element_size) {
  float max_time = 0.0f;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    auto chunk_bytes = splits[i] * element_size;
    float time = paths[i].TransferTimeUsPipelined(chunk_bytes);
    max_time = std::max(max_time, time);
  }
  return max_time;
}

/// Compute per-path element counts from explicit fractional weights.
///
/// Same rounding strategy as ComputeSplits: floor(weight_i * total),
/// remainder assigned to the largest-weight path.
static std::vector<std::size_t> ComputeSplitsFromWeights(
    const std::vector<float>& weights, std::size_t total_elements) {
  ASSERT_VALID_ARGUMENTS(!weights.empty(),
                         "ComputeSplitsFromWeights requires >= 1 weight");

  std::vector<std::size_t> splits(weights.size());
  std::size_t assigned = 0;
  std::size_t best_idx = 0;
  float best_weight = 0.0f;

  for (std::size_t i = 0; i < weights.size(); ++i) {
    splits[i] = static_cast<std::size_t>(static_cast<double>(weights[i]) *
                                         static_cast<double>(total_elements));
    assigned += splits[i];
    if (weights[i] > best_weight) {
      best_weight = weights[i];
      best_idx = i;
    }
  }

  splits[best_idx] += total_elements - assigned;

  return splits;
}

/// Emit a multi-hop copy chain for a single path and buffer chunk.
///
/// For a direct path (2 hops, e.g. dev0 → dev1), emits a single CopyOp.
/// For a multi-hop path (e.g. dev0 → dev2 → dev1), allocates a temporary
/// buffer at each intermediate hop and chains copies through them:
///   AllocTmp(dev2, chunk_size) → Copy(src, tmp) → Copy(tmp, dst)
///
/// Returns the final dst_out value (the consumed destination after the last
/// copy), which the caller uses to map the original CopyOp's output.
static cir::Value EmitCopyChain(cir::ProgramRewriter& rw, const Path& path,
                                cir::Value src_chunk, cir::Value dst_chunk,
                                std::size_t chunk_elements,
                                torch::Dtype dtype) {
  if (path.hops.size() <= 2) {
    return rw.Target().EmitCopy(src_chunk, dst_chunk);
  }

  std::vector<cir::Value> tmps;
  tmps.reserve(path.hops.size() - 2);
  for (const auto& hop :
       std::span(path.hops.begin() + 1, path.hops.end() - 1)) {
    tmps.emplace_back(rw.Target().EmitAllocTmp(hop, chunk_elements, dtype));
  }

  cir::Value prev = src_chunk;
  for (std::size_t j = 0; j < tmps.size(); ++j) {
    prev = rw.Target().EmitCopy(prev, tmps[j]);
  }
  return rw.Target().EmitCopy(prev, dst_chunk);
}

//==============================================================================
// Pass implementation
//==============================================================================

cir::Program BandwidthAggregation::Run(cir::Program program,
                                       const PassContext& ctx) {
  const auto& hints = ctx.hints;
  if (!topo_ && hints.GetHints<RoutingHint>().empty() &&
      hints.GetHints<BandwidthHint>().empty()) {
    return program;
  }

  // Build unified override map.  RoutingHints are normalized to single-path
  // BandwidthHints; actual BandwidthHints overwrite (higher precedence).
  std::map<std::pair<Participant, Participant>, BandwidthHint> overrides;
  for (const auto& ref : hints.GetHints<RoutingHint>()) {
    const auto& h = ref.get();
    overrides.insert_or_assign(std::pair{h.src, h.dst},
                               BandwidthHint(h.src, h.dst, {h.path}, {1.0f}));
  }
  for (const auto& ref : hints.GetHints<BandwidthHint>()) {
    const auto& h = ref.get();
    overrides.insert_or_assign(std::pair{h.src, h.dst}, h);
  }

  auto rw = cir::ProgramRewriter(program);
  for (std::size_t i = 0; i < program.NumOperations(); ++i) {
    const auto& op = program.Operations()[i];
    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;
          if constexpr (std::is_same_v<T, cir::CopyOp>) {
            auto src_val_info = program.GetValueInfo(concrete.src);
            auto dst_val_info = program.GetValueInfo(concrete.dst_in);
            auto bytes = src_val_info.NumBytes();
            auto num_elements = src_val_info.size_elements;
            auto dt = src_val_info.dtype;
            auto element_size = torch::elementSize(dt);

            if (src_val_info.device == dst_val_info.device) {
              rw.CloneOp(i);
              return;
            }

            // --- Resolve paths and splits ---
            std::vector<Path> paths;
            std::vector<std::size_t> splits;

            auto override_it =
                overrides.find({src_val_info.device, dst_val_info.device});
            if (override_it != overrides.end()) {
              const auto& hint = override_it->second;
              paths = hint.paths;
              splits = ComputeSplitsFromWeights(hint.weights, num_elements);
            } else if (topo_) {
              // Find edge-disjoint paths incrementally, stopping as soon as
              // adding another path no longer improves estimated transfer time.
              auto path_iter = topo_->EdgeDisjointPaths(
                  src_val_info.device, dst_val_info.device,
                  [bytes](const Link& l) -> float {
                    return l.TransferTimeUs(bytes);
                  });

              float prev_time = std::numeric_limits<float>::max();
              for (std::size_t p = 0; p < max_paths_; ++p) {
                auto next = path_iter.Next();
                if (!next.has_value()) {
                  break;
                }
                paths.push_back(std::move(*next));
                splits = ComputeSplits(paths, num_elements);
                float time = EstimateTransferTime(paths, splits, element_size);
                if (time > prev_time) {
                  paths.pop_back();
                  splits = ComputeSplits(paths, num_elements);
                  break;
                }
                prev_time = time;
              }

              ASSERT_VALID_RUNTIME(!paths.empty(),
                                   "No path exists between {} and {}",
                                   src_val_info.device, dst_val_info.device);
            } else {
              rw.CloneOp(i);
              return;
            }

            // --- Emit copies along resolved paths ---
            if (paths.size() == 1 && paths[0].hops.size() <= 2) {
              rw.CloneOp(i);
              return;
            }

            auto src = rw.Lookup(concrete.src);
            auto dst_in = rw.Lookup(concrete.dst_in);

            if (paths.size() == 1) {
              auto new_dst_out =
                  EmitCopyChain(rw, paths[0], src, dst_in, num_elements, dt);
              rw.MapValue(concrete.dst_out, new_dst_out);
              return;
            }

            std::size_t offset = 0;
            for (std::size_t p = 0; p < paths.size(); ++p) {
              auto chunk_elements = splits[p];
              if (chunk_elements == 0) {
                continue;
              }

              auto src_slice = rw.Target().EmitSlice(
                  src, cir::Slice{offset, chunk_elements});
              auto dst_slice = rw.Target().EmitSlice(
                  dst_in, cir::Slice{offset, chunk_elements});

              EmitCopyChain(rw, paths[p], src_slice, dst_slice, chunk_elements,
                            dt);

              offset += chunk_elements;
            }

            auto new_dst_out = rw.Target().EmitConsume(dst_in);
            rw.MapValue(concrete.dst_out, new_dst_out);
            return;
          }

          rw.CloneOp(i);
        },
        op.op);
  }
  return rw.Finish();
}

}  // namespace setu::planner::passes
