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

/// Emits CIR for the AllGather replication strategy.
///
/// Each of N replicas copies 1/N of the source, then an AllGather broadcasts
/// the full data to all replicas.
static cir::Program EmitAllGatherStrategy(const CopySpec& copy_spec,
                                          const TensorMetadataPtr& src_meta,
                                          const TensorMetadataPtr& dst_meta) {
  auto num_src_replicas = static_cast<std::size_t>(src_meta->num_replicas);
  auto num_dst_replicas = static_cast<std::size_t>(dst_meta->num_replicas);

  auto total_elements = src_meta->size;
  auto chunk_size = total_elements / num_dst_replicas;

  ASSERT_VALID_RUNTIME(
      total_elements % num_dst_replicas == 0,
      "AllGather strategy requires src elements ({}) to be divisible "
      "by num_dst_replicas ({}). Consider using Naive strategy.",
      total_elements, num_dst_replicas);

  cir::Program program;

  // Collect per-replica info for the AllGather.
  struct ReplicaInfo {
    TensorOwnershipMapPtr src_own;
    TensorShardRangeView src_view;
    TensorOwnershipMapPtr dst_own;
    TensorShardRangeView dst_view;
  };
  std::vector<ReplicaInfo> replicas;
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

    // TODO: Support multi-shard replicas by decomposing into per-shard-position
    // AllGathers. For now, require single shard per replica.
    ASSERT_VALID_RUNTIME(
        dst_own->GetNumShards() == 1,
        "AllGather strategy currently requires single shard per replica, "
        "but replica {} has {} shards. Consider using Naive strategy.",
        r, dst_own->GetNumShards());

    replicas.push_back(ReplicaInfo{.src_own = src_own,
                                   .src_view = TensorShardRangeView(src_own),
                                   .dst_own = dst_own,
                                   .dst_view = TensorShardRangeView(dst_own)});
  }

  // Step 1: Copy 1/N of source to each replica's chunk region.
  // Replica r gets src elements [r*chunk_size, (r+1)*chunk_size) written
  // to dst_replica_r at offset [r*chunk_size, (r+1)*chunk_size).
  for (std::size_t r = 0; r < num_dst_replicas; ++r) {
    auto global_offset = r * chunk_size;
    EmitTwoPointerCopies(program, replicas[r].src_view, replicas[r].dst_view,
                         global_offset, chunk_size);
  }

  // Step 2: Emit AllGather across all replicas.
  // For each replica r:
  //   src  = view of dst_replica_r at [r*chunk_size, chunk_size]  (send region)
  //   dst_in = view of dst_replica_r at [0, total_elements]       (full buffer)
  std::vector<Value> allgather_srcs;
  std::vector<Value> allgather_dst_ins;
  allgather_srcs.reserve(num_dst_replicas);
  allgather_dst_ins.reserve(num_dst_replicas);

  for (std::size_t r = 0; r < num_dst_replicas; ++r) {
    const auto& shard_meta = replicas[r].dst_own->shard_mapping[0].second;
    auto device = Device(shard_meta->owner, shard_meta->spec.device);
    auto shard_ref =
        ref::ShardRef(shard_meta->id, shard_meta->spec.name, shard_meta->owner);
    auto dtype = shard_meta->spec.dtype;

    auto src_val = program.EmitView(
        device, shard_ref, Slice{.offset = r * chunk_size, .size = chunk_size},
        dtype);
    auto dst_val = program.EmitView(
        device, shard_ref, Slice{.offset = 0, .size = total_elements}, dtype);

    allgather_srcs.push_back(src_val);
    allgather_dst_ins.push_back(dst_val);
  }

  (void)program.EmitAllGather(std::move(allgather_srcs),
                              std::move(allgather_dst_ins));

  LOG_DEBUG(
      "AllGather strategy: {} src replicas, {} dst replicas, chunk_size={}, "
      "total={}",
      num_src_replicas, num_dst_replicas, chunk_size, total_elements);

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
  return EmitNaiveStrategy(copy_spec, src_meta, dst_meta);
}

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
