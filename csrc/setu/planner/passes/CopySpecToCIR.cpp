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
#include "planner/passes/CopySpecToCIR.h"
//==============================================================================
#include "commons/Logging.h"
#include "planner/TensorShardRangeView.h"
#include "planner/hints/Hint.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================

using setu::metastore::datatypes::TensorMetadataPtr;
using setu::metastore::datatypes::TensorOwnershipMapPtr;
using setu::planner::ShardBufferRange;
using setu::planner::TensorShardRangeView;
using setu::planner::hints::ReplicationHint;
using setu::planner::hints::ReplicationStrategy;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::Value;
namespace ref = setu::planner::ir::ref;

//==============================================================================

/// Tracks consumption progress within a single shard buffer range.
struct ShardBufferState {
  explicit ShardBufferState(ShardBufferRange range_param)
      : buf(range_param), consumed(0) {}

  [[nodiscard]] bool IsConsumed() const { return consumed == buf.range.length; }

  [[nodiscard]] std::size_t Remaining() const {
    return buf.range.length - consumed;
  }

  void Consume(std::size_t sz) {
    ASSERT_VALID_ARGUMENTS(
        sz <= Remaining(),
        "Tried to consume more than what the buffer has left");
    consumed += sz;
  }

  /// Current offset in elements from the start of the shard buffer.
  [[nodiscard]] std::size_t CurrentOffsetElements() const {
    return buf.range.start + consumed;
  }

  ShardBufferRange buf;
  std::size_t consumed;
};

//==============================================================================

/// Advances to the next shard buffer range if current one is consumed.
static void AdvanceIfConsumed(ShardBufferState& state,
                              TensorShardRangeView::Iterator& it,
                              TensorShardRangeView::Iterator end) {
  if (state.IsConsumed()) {
    ++it;
    if (it != end) {
      state = ShardBufferState(*it);
    }
  }
}

//==============================================================================

/// Two-pointer walk emitting View+Copy CIR ops for matched src/dst regions.
///
/// Optionally windows to a sub-range: skips `global_offset` elements from both
/// src and dst, then copies exactly `global_length` elements.
///
/// When global_offset=0 and global_length=max, copies everything (default).
static void EmitTwoPointerCopies(
    cir::Program& program, const TensorShardRangeView& src_view,
    const TensorShardRangeView& dst_view, std::size_t global_offset = 0,
    std::size_t global_length = std::numeric_limits<std::size_t>::max()) {
  ASSERT_VALID_RUNTIME(!src_view.empty() && !dst_view.empty(),
                       "Source and destination views must not be empty");

  auto src_it = src_view.begin();
  auto dst_it = dst_view.begin();
  auto src = ShardBufferState(*src_it);
  auto dst = ShardBufferState(*dst_it);

  // Skip global_offset elements from both src and dst.
  std::size_t skipped = 0;
  while (skipped < global_offset && src_it != src_view.end() &&
         dst_it != dst_view.end()) {
    auto to_skip =
        std::min({global_offset - skipped, src.Remaining(), dst.Remaining()});
    src.Consume(to_skip);
    dst.Consume(to_skip);
    skipped += to_skip;
    AdvanceIfConsumed(src, src_it, src_view.end());
    AdvanceIfConsumed(dst, dst_it, dst_view.end());
  }

  // Copy global_length elements.
  std::size_t copied = 0;
  while (copied < global_length && src_it != src_view.end() &&
         dst_it != dst_view.end()) {
    auto to_copy =
        std::min({global_length - copied, src.Remaining(), dst.Remaining()});

    auto src_device =
        Device(src.buf.metadata->owner, src.buf.metadata->spec.device);
    auto dst_device =
        Device(dst.buf.metadata->owner, dst.buf.metadata->spec.device);
    auto src_shard_ref =
        ref::ShardRef(src.buf.metadata->id, src.buf.metadata->spec.name,
                      src.buf.metadata->owner);
    auto dst_shard_ref =
        ref::ShardRef(dst.buf.metadata->id, dst.buf.metadata->spec.name,
                      dst.buf.metadata->owner);
    auto dtype = src.buf.metadata->spec.dtype;

    auto src_val = program.EmitView(
        src_device, src_shard_ref,
        Slice{.offset = src.CurrentOffsetElements(), .size = to_copy}, dtype);
    auto dst_val = program.EmitView(
        dst_device, dst_shard_ref,
        Slice{.offset = dst.CurrentOffsetElements(), .size = to_copy}, dtype);
    (void)program.EmitCopy(src_val, dst_val);

    src.Consume(to_copy);
    dst.Consume(to_copy);
    copied += to_copy;
    AdvanceIfConsumed(src, src_it, src_view.end());
    AdvanceIfConsumed(dst, dst_it, dst_view.end());
  }

  if (global_length != std::numeric_limits<std::size_t>::max()) {
    ASSERT_VALID_RUNTIME(copied == global_length,
                         "Only copied {} of {} requested elements", copied,
                         global_length);
  }
}

//==============================================================================

/// Pairing between a source replica and a destination replica.
struct ReplicaPair {
  TensorOwnershipMapPtr src_own;
  TensorShardRangeView src_view;
  TensorOwnershipMapPtr dst_own;
  TensorShardRangeView dst_view;
};

/// Pair destination replicas to source replicas with round-robin src
/// assignment, returning their ownership maps and shard-range views.
static std::vector<ReplicaPair> PairReplicas(
    const CopySpec& copy_spec, const TensorMetadataPtr& src_meta,
    const TensorMetadataPtr& dst_meta) {
  auto num_src_replicas = static_cast<std::size_t>(src_meta->num_replicas);
  auto num_dst_replicas = static_cast<std::size_t>(dst_meta->num_replicas);

  std::vector<ReplicaPair> replicas;
  replicas.reserve(num_dst_replicas);

  for (std::size_t r = 0; r < num_dst_replicas; ++r) {
    // round-robin across sources
    auto src_replica_id =
        static_cast<setu::commons::ReplicaId>(r % num_src_replicas);
    auto dst_replica_id = static_cast<setu::commons::ReplicaId>(r);
    auto src_own = src_meta->GetOwnershipMapForReplica(copy_spec.src_selection,
                                                       src_replica_id);
    auto dst_own = dst_meta->GetOwnershipMapForReplica(copy_spec.dst_selection,
                                                       dst_replica_id);
    auto src_view = TensorShardRangeView(src_own);
    auto dst_view = TensorShardRangeView(dst_own);

    replicas.push_back(ReplicaPair{.src_own = std::move(src_own),
                                   .src_view = std::move(src_view),
                                   .dst_own = std::move(dst_own),
                                   .dst_view = std::move(dst_view)});
  }

  return replicas;
}

//==============================================================================

/// Emits CIR for the AllGather replication strategy.
///
/// Each of N replicas copies 1/N of the source, then an AllGather broadcasts
/// the full data to all replicas.
static cir::Program EmitAllGatherStrategy(const CopySpec& copy_spec,
                                          const TensorMetadataPtr& src_meta,
                                          const TensorMetadataPtr& dst_meta) {
  auto num_src_replicas = static_cast<std::size_t>(src_meta->num_replicas);
  auto num_dst_replicas = static_cast<std::size_t>(dst_meta->num_replicas);

  cir::Program program;

  auto replicas = PairReplicas(copy_spec, src_meta, dst_meta);

  // We derive the number of contiguous pieces from the view of the first
  // replica. Each replica has the same sharding strategy, so this is correct.
  auto num_pieces = replicas[0].dst_view.size();
  ASSERT_VALID_RUNTIME(num_pieces > 0,
                       "AllGather strategy requires a non-empty dst view");

  // Emit one sparse pull + AllGather per piece.
  std::size_t piece_selection_offset = 0;
  std::size_t total_selection_elements = 0;
  for (std::size_t i = 0; i < num_pieces; ++i) {
    auto piece_size = (replicas[0].dst_view.begin() + i)->range.length;

    ASSERT_VALID_RUNTIME(
        piece_size > 0 && piece_size % num_dst_replicas == 0,
        "AllGather strategy requires piece {} size ({}) to be positive "
        "and divisible by num_dst_replicas ({}). Consider using Naive "
        "strategy.",
        i, piece_size, num_dst_replicas);

    auto chunk_size = piece_size / num_dst_replicas;

    // Step 1: scatter piece i. Each dst replica r pulls chunk r of this
    // piece from its round-robin src replica. The two-pointer walk handles
    // src-side multi-range naturally.
    for (std::size_t r = 0; r < num_dst_replicas; ++r) {
      auto global_offset = piece_selection_offset + r * chunk_size;
      EmitTwoPointerCopies(program, replicas[r].src_view, replicas[r].dst_view,
                           global_offset, chunk_size);
    }

    // Step 2: AllGather for piece i
    std::vector<Value> allgather_srcs;
    std::vector<Value> allgather_dst_ins;
    allgather_srcs.reserve(num_dst_replicas);
    allgather_dst_ins.reserve(num_dst_replicas);
    for (std::size_t r = 0; r < num_dst_replicas; ++r) {
      const auto& piece = replicas[r].dst_view.begin() + i;
      const auto& shard_meta = piece->metadata;
      auto device = Device(shard_meta->owner, shard_meta->spec.device);
      auto shard_ref = ref::ShardRef(shard_meta->id, shard_meta->spec.name,
                                     shard_meta->owner);
      auto dtype = shard_meta->spec.dtype;
      auto piece_base = piece->range.start;
      auto src_val = program.EmitView(
          device, shard_ref,
          Slice{.offset = piece_base + r * chunk_size, .size = chunk_size},
          dtype);
      auto dst_val = program.EmitView(
          device, shard_ref,
          Slice{.offset = piece_base, .size = piece_size}, dtype);
      allgather_srcs.push_back(src_val);
      allgather_dst_ins.push_back(dst_val);
    }
    (void)program.EmitAllGather(std::move(allgather_srcs),
                                std::move(allgather_dst_ins));
    piece_selection_offset += piece_size;
    total_selection_elements += piece_size;
  }

  LOG_DEBUG(
      "AllGather strategy: {} src replicas, {} dst replicas, "
      "{} pieces, selection_size={}",
      num_src_replicas, num_dst_replicas, num_pieces, total_selection_elements);

  return program;
}

//==============================================================================

/// Emits CIR for the BatchedCopy replication strategy.
///
/// Each of N destination replicas pulls 1/N of the source into its own chunk
/// slot within its destination buffer, then N*(N-1) inter-destination copies
/// propagate every chunk across all destination replicas.
///
/// Unlike the AllGather strategy this does not need piece-level decomposition
/// or rank-ordering: EmitTwoPointerCopies transparently handles multi-range
/// and multi-shard src/dst views on both sides.
static cir::Program EmitBatchedCopyStrategy(const CopySpec& copy_spec,
                                            const TensorMetadataPtr& src_meta,
                                            const TensorMetadataPtr& dst_meta) {
  auto num_src_replicas = static_cast<std::size_t>(src_meta->num_replicas);
  auto num_dst_replicas = static_cast<std::size_t>(dst_meta->num_replicas);

  cir::Program program;

  auto replicas = PairReplicas(copy_spec, src_meta, dst_meta);

  // Selection size is the same across replicas (each replica holds a full
  // copy of the selection).
  std::size_t selection_size = 0;
  for (const auto& shard_range : replicas[0].src_view) {
    selection_size += shard_range.range.length;
  }
  ASSERT_VALID_RUNTIME(
      selection_size > 0 && selection_size % num_dst_replicas == 0,
      "BatchedCopy strategy requires selection elements ({}) to be positive "
      "and divisible by num_dst_replicas ({}). Consider using Naive "
      "strategy.",
      selection_size, num_dst_replicas);
  auto chunk_size = selection_size / num_dst_replicas;

  // Step 1: scatter. Each dst replica r pulls chunk r from its round-robin
  // src replica into its own dst view at selection offset
  // [r*chunk_size, (r+1)*chunk_size).
  for (std::size_t r = 0; r < num_dst_replicas; ++r) {
    EmitTwoPointerCopies(program, replicas[r].src_view, replicas[r].dst_view,
                         r * chunk_size, chunk_size);
  }

  // Step 2: batched gather among destination replicas. For every ordered
  // (k, j) pair with k != j, copy chunk k from dst replica k to dst replica
  // j at the same selection offset.
  for (std::size_t k = 0; k < num_dst_replicas; ++k) {
    for (std::size_t j = 0; j < num_dst_replicas; ++j) {
      if (k == j) continue;
      EmitTwoPointerCopies(program, replicas[k].dst_view, replicas[j].dst_view,
                           k * chunk_size, chunk_size);
    }
  }

  LOG_DEBUG(
      "BatchedCopy strategy: {} src replicas, {} dst replicas, "
      "selection_size={}, chunk_size={}",
      num_src_replicas, num_dst_replicas, selection_size, chunk_size);

  return program;
}

//==============================================================================

/// Emits CIR for the Naive replication strategy.
///
/// Each replica independently copies the full source.
static cir::Program EmitNaiveStrategy(const CopySpec& copy_spec,
                                      const TensorMetadataPtr& src_meta,
                                      const TensorMetadataPtr& dst_meta) {
  auto num_src_replicas = static_cast<std::size_t>(src_meta->num_replicas);
  auto num_dst_replicas = static_cast<std::size_t>(dst_meta->num_replicas);

  cir::Program program;

  for (std::size_t r = 0; r < num_dst_replicas; ++r) {
    // round-robin across sources
    auto src_replica_id =
        static_cast<setu::commons::ReplicaId>(r % num_src_replicas);
    auto dst_replica_id = static_cast<setu::commons::ReplicaId>(r);
    auto src_own = src_meta->GetOwnershipMapForReplica(copy_spec.src_selection,
                                                       src_replica_id);
    auto dst_own = dst_meta->GetOwnershipMapForReplica(copy_spec.dst_selection,
                                                       dst_replica_id);
    auto dst_view = TensorShardRangeView(dst_own);
    auto src_view = TensorShardRangeView(src_own);

    EmitTwoPointerCopies(program, src_view, dst_view);
  }

  LOG_DEBUG(
      "Naive strategy: {} src replicas, {} dst replicas, {} elements each",
      num_src_replicas, num_dst_replicas, src_meta->size);

  return program;
}

//==============================================================================

cir::Program CopySpecToCIR::Run(const CopySpec& copy_spec, MetaStore& metastore,
                                const HintStore& hints) {
  auto src_meta = metastore.GetTensorMetadata(copy_spec.src_name);
  auto dst_meta = metastore.GetTensorMetadata(copy_spec.dst_name);

  // Non-replicated destination: existing two-pointer algorithm.
  if (dst_meta->num_replicas == 1) {
    auto src_own = src_meta->GetOwnershipMap(copy_spec.src_selection);
    auto dst_own = dst_meta->GetOwnershipMap(copy_spec.dst_selection);
    auto src_view = TensorShardRangeView(src_own);
    auto dst_view = TensorShardRangeView(dst_own);

    cir::Program program;
    EmitTwoPointerCopies(program, src_view, dst_view);

    ASSERT_VALID_RUNTIME(src_view.size() > 0 && dst_view.size() > 0,
                         "Source and destination views must not be empty");

    return program;
  }

  // Replicated destination: determine strategy from hints.
  auto strategy = ReplicationStrategy::kAllGather;
  auto repl_hints = hints.GetHints<ReplicationHint>();
  for (const auto& hint : repl_hints) {
    if (hint.get().dst_name == copy_spec.dst_name) {
      strategy = hint.get().strategy;
      LOG_DEBUG("Using {} strategy for replicated tensor {} (from hint)",
                hint.get().ToString(), copy_spec.dst_name);
      break;
    }
  }

  if (strategy == ReplicationStrategy::kAllGather) {
    return EmitAllGatherStrategy(copy_spec, src_meta, dst_meta);
  }
  if (strategy == ReplicationStrategy::kBatchedCopy) {
    return EmitBatchedCopyStrategy(copy_spec, src_meta, dst_meta);
  }
  return EmitNaiveStrategy(copy_spec, src_meta, dst_meta);
}

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
