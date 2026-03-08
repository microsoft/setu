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
#include <nccl.h>
//==============================================================================
#include "commons/Logging.h"
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Operation.h"
#include "planner/ir/llc/Instruction.h"
#include "planner/ir/ref/BufferRef.h"
#include "planner/ir/ref/RegisterRef.h"
//==============================================================================
namespace setu::planner::targets {
//==============================================================================

namespace llc = setu::planner::ir::llc;
namespace ref = setu::planner::ir::ref;

//==============================================================================

/// Per-value metadata captured when lowering ViewOp, AllocTmpOp, or SliceOp.
struct ViewInfo {
  Participant participant;
  ref::BufferRef buffer_ref;
  std::size_t offset_bytes;
  std::size_t count;
  torch::Dtype dtype;
};

/// Intermediate copy between two views, collected before LLC emission.
struct PendingCopy {
  Participant src_part;
  ref::BufferRef src_ref;
  std::size_t src_offset_bytes;

  Participant dst_part;
  ref::BufferRef dst_ref;
  std::size_t dst_offset_bytes;

  std::size_t count;
  torch::Dtype dtype;

  std::uint32_t cir_op_index;  ///< Index in the CIR program, for depth lookup
};

//==============================================================================

NCCL::NCCL(
    std::unordered_map<cir::Device, setu::planner::RegisterSet> register_sets)
    : register_sets_(std::move(register_sets)) {}

void NCCL::AddRegisterSets(
    const std::unordered_map<cir::Device, setu::planner::RegisterSet>&
        register_sets) {
  for (const auto& [device, reg_set] : register_sets) {
    register_sets_.insert_or_assign(device, reg_set);
  }
}

//==============================================================================

Plan NCCL::Run(const cir::Program& program) {
  std::unordered_map<cir::Value, ViewInfo> view_map;

  Plan plan;
  Participants& parts = plan.participants;
  std::vector<PendingCopy> pending_copies;

  // === Step 0: Register allocation (only if AllocTmpOps are present) ===

  bool has_alloc_tmp = std::ranges::any_of(
      program.Operations(),
      [](const auto& op) { return op.Type() == cir::OpType::kAllocTmp; });

  std::optional<cir::RegisterAllocation> reg_alloc;
  if (has_alloc_tmp) {
    auto liveness = cir::LivenessInfo::Build(program);
    reg_alloc =
        cir::RegisterAllocation::Build(program, liveness, register_sets_);
  }

  // === Step 1: Walk CIR ops, collect view info and pending copies ===

  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;

          if constexpr (std::is_same_v<T, cir::ViewOp>) {
            auto element_size = torch::elementSize(concrete.dtype);
            auto offset_bytes = concrete.slice.offset * element_size;

            ViewInfo info{
                .participant = concrete.device,
                .buffer_ref = ref::BufferRef(concrete.handle),
                .offset_bytes = offset_bytes,
                .count = concrete.slice.size,
                .dtype = concrete.dtype,
            };

            parts.insert(concrete.device);
            view_map.try_emplace(concrete.out, std::move(info));

          } else if constexpr (std::is_same_v<T, cir::AllocTmpOp>) {
            ASSERT_VALID_RUNTIME(
                reg_alloc.has_value() &&
                    reg_alloc->allocation[concrete.out.id].has_value(),
                "AllocTmpOp {} has no register allocation",
                concrete.out.ToString());

            const auto& phys_reg =
                reg_alloc->allocation[concrete.out.id].value();

            ViewInfo info{
                .participant = concrete.device,
                .buffer_ref = ref::BufferRef(
                    ref::RegisterRef(phys_reg.register_index, concrete.device)),
                .offset_bytes = 0,
                .count = concrete.size_elements,
                .dtype = concrete.dtype,
            };

            parts.insert(concrete.device);
            view_map.try_emplace(concrete.out, std::move(info));

          } else if constexpr (std::is_same_v<T, cir::SliceOp>) {
            auto src_it = view_map.find(concrete.src);
            ASSERT_VALID_RUNTIME(src_it != view_map.end(),
                                 "SliceOp source {} not found in view_map",
                                 concrete.src.ToString());

            const auto& src_info = src_it->second;
            auto element_size = torch::elementSize(src_info.dtype);

            ViewInfo info{
                .participant = src_info.participant,
                .buffer_ref = src_info.buffer_ref,
                .offset_bytes = src_info.offset_bytes +
                                concrete.slice.offset * element_size,
                .count = concrete.slice.size,
                .dtype = src_info.dtype,
            };

            view_map.try_emplace(concrete.out, std::move(info));

          } else if constexpr (std::is_same_v<T, cir::CopyOp>) {
            auto src_it = view_map.find(concrete.src);
            auto dst_it = view_map.find(concrete.dst_in);
            ASSERT_VALID_RUNTIME(
                src_it != view_map.end() && dst_it != view_map.end(),
                "CopyOp operands {} and {} must be resolvable in view_map",
                concrete.src.ToString(), concrete.dst_in.ToString());

            const auto& src = src_it->second;
            const auto& dst = dst_it->second;

            pending_copies.push_back(PendingCopy{
                .src_part = src.participant,
                .src_ref = src.buffer_ref,
                .src_offset_bytes = src.offset_bytes,
                .dst_part = dst.participant,
                .dst_ref = dst.buffer_ref,
                .dst_offset_bytes = dst.offset_bytes,
                .count = src.count,
                .dtype = src.dtype,
                .cir_op_index = op_idx,
            });

            // dst_out inherits dst view info so downstream copies resolve.
            view_map.try_emplace(concrete.dst_out, dst_it->second);

          } else if constexpr (std::is_same_v<T, cir::ConsumeOp>) {
            // Consume is a marker op; propagate view info from src to out
            auto src_it = view_map.find(concrete.src);
            ASSERT_VALID_RUNTIME(src_it != view_map.end(),
                                 "ConsumeOp source {} not found in view_map",
                                 concrete.src.ToString());
            view_map.try_emplace(concrete.out, src_it->second);

          } else if constexpr (std::is_same_v<T, cir::PackOp>) {
            auto dst_it = view_map.find(concrete.dst_in);
            ASSERT_VALID_RUNTIME(dst_it != view_map.end(),
                                 "PackOp dst_in {} not found in view_map",
                                 concrete.dst_in.ToString());

            const auto& dst = dst_it->second;

            // Each source is copied into the destination at a running offset.
            // Sources are packed contiguously in order.
            std::size_t running_offset_bytes = dst.offset_bytes;
            for (const auto& src_val : concrete.srcs) {
              auto src_it = view_map.find(src_val);
              ASSERT_VALID_RUNTIME(src_it != view_map.end(),
                                   "PackOp source {} not found in view_map",
                                   src_val.ToString());

              const auto& src = src_it->second;

              pending_copies.push_back(PendingCopy{
                  .src_part = src.participant,
                  .src_ref = src.buffer_ref,
                  .src_offset_bytes = src.offset_bytes,
                  .dst_part = dst.participant,
                  .dst_ref = dst.buffer_ref,
                  .dst_offset_bytes = running_offset_bytes,
                  .count = src.count,
                  .dtype = src.dtype,
                  .cir_op_index = op_idx,
              });

              running_offset_bytes += src.count * torch::elementSize(src.dtype);
            }

            // dst_out inherits dst view info so downstream ops resolve.
            view_map.try_emplace(concrete.dst_out, dst_it->second);

          } else if constexpr (std::is_same_v<T, cir::UnpackOp>) {
            auto src_it = view_map.find(concrete.src);
            ASSERT_VALID_RUNTIME(src_it != view_map.end(),
                                 "UnpackOp source {} not found in view_map",
                                 concrete.src.ToString());

            const auto& src = src_it->second;

            ASSERT_VALID_RUNTIME(
                concrete.dst_ins.size() == concrete.dst_outs.size(),
                "UnpackOp dst_ins and dst_outs size mismatch");

            // Each destination receives a contiguous slice of the source.
            // Destinations are filled in order.
            std::size_t running_offset_bytes = src.offset_bytes;
            for (std::size_t i = 0; i < concrete.dst_ins.size(); ++i) {
              auto dst_it = view_map.find(concrete.dst_ins[i]);
              ASSERT_VALID_RUNTIME(dst_it != view_map.end(),
                                   "UnpackOp dst_in {} not found in view_map",
                                   concrete.dst_ins[i].ToString());

              const auto& dst = dst_it->second;

              pending_copies.push_back(PendingCopy{
                  .src_part = src.participant,
                  .src_ref = src.buffer_ref,
                  .src_offset_bytes = running_offset_bytes,
                  .dst_part = dst.participant,
                  .dst_ref = dst.buffer_ref,
                  .dst_offset_bytes = dst.offset_bytes,
                  .count = dst.count,
                  .dtype = dst.dtype,
                  .cir_op_index = op_idx,
              });

              running_offset_bytes += dst.count * torch::elementSize(dst.dtype);

              // Each dst_out inherits its corresponding dst_in view info.
              view_map.try_emplace(concrete.dst_outs[i], dst_it->second);
            }

          } else {
            RAISE_RUNTIME_ERROR("NCCL backend: unsupported CIR operation");
          }
        },
        op.op);
  }

  ASSERT_VALID_RUNTIME(!parts.empty(), "No participants found in CIR program");

  // === Step 2: Set up per-pair communicators ===
  //
  // Create a separate 2-GPU communicator for each unique (src, dst)
  // participant pair.  This prevents NCCL proxy thread serialization that
  // occurs when multiple independent Send/Recv ops share one communicator.

  auto& programs = plan.program;

  // Collect unique cross-device participant pairs
  std::set<Participants> unique_pair_parts;
  for (const auto& c : pending_copies) {
    if (c.src_part != c.dst_part) {
      Participants pair_parts;
      pair_parts.insert(c.src_part);
      pair_parts.insert(c.dst_part);
      unique_pair_parts.insert(pair_parts);
    }
  }

  // Create and emit InitComm for each pair in deterministic order.
  // ncclCommInitRank is collective, so both sides must call it.  The
  // deterministic ordering of unique_pair_parts (std::set) plus the fact
  // that each device only sees its own InitComms prevents deadlock.
  for (const auto& pair_parts : unique_pair_parts) {
    if (!comm_cache_.contains(pair_parts)) {
      ncclUniqueId nccl_id;
      ncclGetUniqueId(&nccl_id);
      auto comm_id = CommId::From(nccl_id);

      DeviceRank rank = 0;
      std::unordered_map<Participant, DeviceRank> ranks;
      for (const auto& p : pair_parts) {
        ranks[p] = rank++;
      }
      comm_cache_[pair_parts] = CommCacheEntry{.id = comm_id, .ranks = ranks};

      for (const auto& p : pair_parts) {
        programs[p].emplace_back(llc::InitComm(comm_id, ranks));
      }
    }
  }

  // === Step 3: Data-dependency-driven emission ===
  //
  // Emit copies in CIR program order (no sorting by depth).  Insert Fence
  // instructions only where a participant would read from a buffer region
  // that a prior instruction wrote to on the same device — i.e., where a
  // cross-stream data dependency exists.
  //
  // TODO(future): Fence is a full cross-stream sync (coarser than necessary).
  // The ideal solution is event-based partial sync via new LLC instructions
  // (SyncPoint/WaitSync) mapping to cudaEventRecord/cudaStreamWaitEvent,
  // giving per-dependency sync without global barriers.

  struct WriteRegion {
    ref::BufferRef buffer;
    std::size_t offset;
    std::size_t size;
  };

  // Per-participant: buffer regions written by Receive/Copy that haven't
  // been fenced yet.
  std::unordered_map<Participant, std::vector<WriteRegion>> pending_writes;

  // Check if a read region overlaps any pending write on a participant.
  auto has_conflict = [&](const Participant& part, const ref::BufferRef& buf,
                          std::size_t offset, std::size_t size) -> bool {
    auto it = pending_writes.find(part);
    if (it == pending_writes.end()) return false;
    for (const auto& w : it->second) {
      if (w.buffer == buf && w.offset < offset + size &&
          offset < w.offset + w.size) {
        return true;
      }
    }
    return false;
  };

  // Flush accumulated same-device copies for a single participant.
  std::unordered_map<Participant, std::vector<llc::CopyEntry>> copy_batches;

  auto flush_copy_batches_for = [&](const Participant& part) {
    auto it = copy_batches.find(part);
    if (it != copy_batches.end() && !it->second.empty()) {
      programs[part].emplace_back(llc::Copy(std::move(it->second)));
      it->second.clear();
    }
  };

  auto flush_all_copy_batches = [&]() {
    for (auto& [part, batch] : copy_batches) {
      if (!batch.empty()) {
        programs[part].emplace_back(llc::Copy(std::move(batch)));
      }
    }
    copy_batches.clear();
  };

  // Emit a Fence for a participant and clear its pending writes.
  auto fence_participant = [&](const Participant& part) {
    flush_copy_batches_for(part);
    programs[part].emplace_back(llc::Fence());
    pending_writes[part].clear();
  };

  for (const auto& c : pending_copies) {
    if (c.src_part == c.dst_part) {
      // Local copy: reads src, writes dst — check src for conflicts
      auto read_size = c.count * torch::elementSize(c.dtype);
      if (has_conflict(c.src_part, c.src_ref, c.src_offset_bytes, read_size)) {
        fence_participant(c.src_part);
      }
      copy_batches[c.src_part].emplace_back(c.src_ref, c.src_offset_bytes,
                                            c.dst_ref, c.dst_offset_bytes,
                                            c.count, c.dtype);
      // Record dst as written
      auto write_size = c.count * torch::elementSize(c.dtype);
      pending_writes[c.dst_part].push_back(
          {c.dst_ref, c.dst_offset_bytes, write_size});
    } else {
      // Cross-device: Send reads src on src_part, Receive writes dst on
      // dst_part.

      // Check if the Send's read conflicts with a prior write on src_part
      auto read_size = c.count * torch::elementSize(c.dtype);
      if (has_conflict(c.src_part, c.src_ref, c.src_offset_bytes, read_size)) {
        fence_participant(c.src_part);
      }

      // Look up the per-pair communicator
      Participants pair_parts;
      pair_parts.insert(c.src_part);
      pair_parts.insert(c.dst_part);
      const auto& pair_entry = comm_cache_.at(pair_parts);

      // Flush any batched local copies before emitting Send/Receive
      flush_copy_batches_for(c.src_part);
      flush_copy_batches_for(c.dst_part);

      programs[c.src_part].emplace_back(
          llc::Send(pair_entry.id, c.src_ref, c.src_offset_bytes, c.count,
                    c.dtype, pair_entry.ranks.at(c.dst_part)));
      programs[c.dst_part].emplace_back(
          llc::Receive(pair_entry.id, c.dst_ref, c.dst_offset_bytes, c.count,
                       c.dtype, pair_entry.ranks.at(c.src_part)));

      // Record the Receive's write on dst_part
      auto write_size = c.count * torch::elementSize(c.dtype);
      pending_writes[c.dst_part].push_back(
          {c.dst_ref, c.dst_offset_bytes, write_size});
    }
  }

  flush_all_copy_batches();

  return plan;
}

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
