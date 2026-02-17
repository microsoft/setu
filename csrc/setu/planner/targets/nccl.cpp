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
#include "planner/targets/nccl.h"
//==============================================================================
#include "commons/Logging.h"
#include "planner/ir/cir/Operation.h"
#include "planner/ir/llc/Instruction.h"
//==============================================================================
namespace setu::planner::targets {
//==============================================================================

namespace llc = setu::planner::ir::llc;

//==============================================================================

/// Per-value metadata captured when lowering a ViewOp.
struct ViewInfo {
  Participant participant;
  llc::ShardRef shard_ref;
  std::size_t offset_bytes;
  std::size_t count;
  torch::Dtype dtype;
};

/// Intermediate copy between two views, collected before LLC emission.
struct PendingCopy {
  Participant src_part;
  llc::ShardRef src_ref;
  std::size_t src_offset_bytes;

  Participant dst_part;
  llc::ShardRef dst_ref;
  std::size_t dst_offset_bytes;

  std::size_t count;
  torch::Dtype dtype;
};

//==============================================================================

Plan NCCL::Run(const cir::Program& program) {
  std::unordered_map<cir::Value, ViewInfo> view_map;

  Plan plan;
  Participants& parts = plan.participants;
  std::vector<PendingCopy> pending_copies;

  // === Step 1: Walk CIR ops, collect view info and pending copies ===

  for (const auto& op : program.Operations()) {
    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;

          if constexpr (std::is_same_v<T, cir::ViewOp>) {
            auto element_size = torch::elementSize(concrete.dtype);
            auto offset_bytes = concrete.slice.offset * element_size;

            ViewInfo info{
                .participant = concrete.device,
                .shard_ref = concrete.handle,
                .offset_bytes = offset_bytes,
                .count = concrete.slice.size,
                .dtype = concrete.dtype,
            };

            parts.insert(concrete.device);
            view_map.try_emplace(concrete.out, std::move(info));

          } else if constexpr (std::is_same_v<T, cir::CopyOp>) {
            auto src_it = view_map.find(concrete.src);
            auto dst_it = view_map.find(concrete.dst_in);
            ASSERT_VALID_RUNTIME(
                src_it != view_map.end() && dst_it != view_map.end(),
                "CopyOp operands must be defined by ViewOps");

            const auto& src = src_it->second;
            const auto& dst = dst_it->second;

            pending_copies.push_back(PendingCopy{
                .src_part = src.participant,
                .src_ref = src.shard_ref,
                .src_offset_bytes = src.offset_bytes,
                .dst_part = dst.participant,
                .dst_ref = dst.shard_ref,
                .dst_offset_bytes = dst.offset_bytes,
                .count = src.count,
                .dtype = src.dtype,
            });

            // dst_out inherits dst view info so downstream copies resolve.
            view_map.try_emplace(concrete.dst_out, dst_it->second);

          } else {
            RAISE_RUNTIME_ERROR(
                "NCCL backend: unsupported CIR operation (only view and copy "
                "are supported)");
          }
        },
        op.op);
  }

  ASSERT_VALID_RUNTIME(!parts.empty(), "No participants found in CIR program");

  // === Step 2: Set up communicator, read from cache if available ===

  bool new_comm = false;
  if (!comm_cache_.contains(parts)) {
    ncclUniqueId comm_id;
    ncclGetUniqueId(&comm_id);

    DeviceRank rank = 0;
    std::unordered_map<Participant, DeviceRank> ranks;
    for (const auto& part : parts) {
      ranks[part] = rank++;
    }
    comm_cache_[parts] = CommCacheEntry{.id = comm_id, .ranks = ranks};
    new_comm = true;
  }

  const auto& entry = comm_cache_.at(parts);

  auto& programs = plan.program;
  for (const auto& part : parts) {
    if (new_comm) {
      programs[part].emplace_back(llc::InitComm(entry.id, entry.ranks));
    } else {
      programs[part].emplace_back(llc::UseComm(entry.id));
    }
  }

  // === Step 3: Emit Copy / Send+Receive for each pending copy ===

  for (const auto& c : pending_copies) {
    if (c.src_part == c.dst_part) {
      programs[c.src_part].emplace_back(llc::Copy(
          c.src_ref, c.src_offset_bytes, c.dst_ref, c.dst_offset_bytes,
          c.count, c.dtype));
    } else {
      programs[c.src_part].emplace_back(llc::Send(
          c.src_ref, c.src_offset_bytes, c.count, c.dtype,
          entry.ranks.at(c.dst_part)));
      programs[c.dst_part].emplace_back(llc::Receive(
          c.dst_ref, c.dst_offset_bytes, c.count, c.dtype,
          entry.ranks.at(c.src_part)));
    }
  }

  return plan;
}

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
