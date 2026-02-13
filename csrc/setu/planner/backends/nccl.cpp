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
#include "planner/backends/nccl.h"

#include "ir/Instruction.h"
#include "planner/TensorShardRangeView.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::planner::backends::nccl {
using setu::commons::DeviceRank;
using setu::ir::ShardRef;
using setu::planner::TensorShardRangeView;
//==============================================================================

// Helper struct to keep track of buffer consumption
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

  [[nodiscard]] std::size_t CurrentOffsetBytes() const {
    auto dtype = buf.metadata->spec.dtype;
    auto element_size = torch::elementSize(dtype);
    return (buf.range.start + consumed) * element_size;
  }

  ShardBufferRange buf;
  std::size_t consumed;
};

// Naive compilation scheme
//
// This scheme emits Send/Recv pairs for each pair of same-sized buffers in src
// and dst. If the buffers reside on the same device, it emits an Copy
// instruction instead.

struct CopyOp {
  const Participant src_part;
  const ShardRef src_ref;
  const std::size_t src_offset_bytes;

  const Participant dst_part;
  const ShardRef dst_ref;
  const std::size_t dst_offset_bytes;

  const std::size_t to_copy;
  torch::Dtype dtype;
};

Plan NCCLPlanner::Compile(CopySpec& copy_spec, MetaStore& metastore) {
  auto src_own = metastore.GetTensorMetadata(copy_spec.src_name)
                     ->GetOwnershipMap(copy_spec.src_selection);
  auto dst_own = metastore.GetTensorMetadata(copy_spec.dst_name)
                     ->GetOwnershipMap(copy_spec.dst_selection);

  auto src_view = TensorShardRangeView(src_own);
  auto src_it = src_view.begin();
  auto dst_view = TensorShardRangeView(dst_own);
  auto dst_it = dst_view.begin();

  ASSERT_VALID_RUNTIME(!src_view.empty() && !dst_view.empty(),
                       "Source and destination views must not be empty");

  Plan plan;
  Participants comm_participants;

  auto src = ShardBufferState(*src_it);
  auto dst = ShardBufferState(*dst_it);

  auto advance_if_consumed = [](ShardBufferState& state, auto& it, auto end) {
    if (state.IsConsumed()) {
      ++it;
      if (it != end) {
        state = ShardBufferState(*it);
      }
    }
  };

  auto& parts = plan.participants;
  auto& program = plan.program;

  std::vector<CopyOp> copy_ops;
  while (src_it != src_view.end() && dst_it != dst_view.end()) {
    auto to_copy = std::min(src.Remaining(), dst.Remaining());
    auto src_part =
        Participant(src.buf.metadata->owner, src.buf.metadata->spec.device);
    auto dst_part =
        Participant(dst.buf.metadata->owner, dst.buf.metadata->spec.device);

    parts.insert(src_part);
    parts.insert(dst_part);

    auto src_shard_ref =
        ir::ShardRef(src.buf.metadata->id, src.buf.metadata->spec.name,
                     src.buf.metadata->owner);
    auto dst_shard_ref =
        ir::ShardRef(dst.buf.metadata->id, dst.buf.metadata->spec.name,
                     dst.buf.metadata->owner);
    auto dtype = src.buf.metadata->spec.dtype;

    copy_ops.emplace_back(CopyOp{.src_part = src_part,
                                 .src_ref = src_shard_ref,
                                 .src_offset_bytes = src.CurrentOffsetBytes(),
                                 .dst_part = dst_part,
                                 .dst_ref = dst_shard_ref,
                                 .dst_offset_bytes = dst.CurrentOffsetBytes(),
                                 .to_copy = to_copy,
                                 .dtype = dtype});

    src.Consume(to_copy);
    dst.Consume(to_copy);
    advance_if_consumed(src, src_it, src_view.end());
    advance_if_consumed(dst, dst_it, dst_view.end());
  }

  ASSERT_VALID_RUNTIME(
      src_it == src_view.end() && dst_it == dst_view.end(),
      "Exited the loop before all src and dst buffers were consumed!");

  ASSERT_VALID_RUNTIME(!parts.empty(), "No participants were discovered");

  bool new_comm = false;
  if (!comm_cache_.contains(parts)) {
    ncclUniqueId comm_id;
    ncclGetUniqueId(&comm_id);
    DeviceRank rank = 0;
    std::unordered_map<Participant, DeviceRank> ranks;
    for (auto part : parts) {
      ranks[part] = rank++;
    }
    comm_cache_[parts] = CommCacheEntry{.id = comm_id, .ranks = ranks};
    new_comm = true;
  }

  auto& entry = comm_cache_.at(parts);

  // Add init/use comm instruction to each participant's program
  for (const auto& part : parts) {
    if (new_comm) {
      program[part].emplace_back(ir::InitComm(entry.id, entry.ranks));
    } else {
      program[part].emplace_back(ir::UseComm(entry.id));
    }
  }

  for (auto op : copy_ops) {
    auto [src_part, src_ref, src_offset_bytes, dst_part, dst_ref,
          dst_offset_bytes, to_copy, dtype] = op;
    if (src_part == dst_part) {
      program[src_part].emplace_back(ir::Copy(src_ref, src_offset_bytes,
                                              dst_ref, dst_offset_bytes,
                                              to_copy, dtype));
    } else {
      program[src_part].emplace_back(ir::Send(
          src_ref, src_offset_bytes, to_copy, dtype, entry.ranks[dst_part]));
      program[dst_part].emplace_back(ir::Receive(
          dst_ref, dst_offset_bytes, to_copy, dtype, entry.ranks[src_part]));
    }
  }

  return plan;
}

//==============================================================================
}  // namespace setu::planner::backends::nccl
//==============================================================================
