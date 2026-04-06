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

/// Per-participant data for a pending AllGather collective.
struct AllGatherParticipant {
  Participant participant;
  ref::BufferRef send_ref;
  std::size_t send_offset_bytes;
  ref::BufferRef recv_ref;
  std::size_t recv_offset_bytes;
};

/// Pending AllGather collective, collected before LLC emission.
struct PendingAllGather {
  std::vector<AllGatherParticipant> participants;
  std::size_t send_count;
  torch::Dtype dtype;
  std::uint32_t cir_op_index;
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
  std::vector<PendingAllGather> pending_all_gathers;

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

          } else if constexpr (std::is_same_v<T, cir::AllGatherOp>) {
            ASSERT_VALID_RUNTIME(
                concrete.srcs.size() == concrete.dst_ins.size() &&
                    concrete.srcs.size() == concrete.dst_outs.size(),
                "AllGatherOp: srcs/dst_ins/dst_outs size mismatch");
            ASSERT_VALID_RUNTIME(concrete.srcs.size() >= 2,
                                 "AllGatherOp: need at least 2 participants");

            PendingAllGather pending;
            pending.cir_op_index = op_idx;

            for (std::size_t i = 0; i < concrete.srcs.size(); ++i) {
              auto src_it = view_map.find(concrete.srcs[i]);
              auto dst_it = view_map.find(concrete.dst_ins[i]);
              ASSERT_VALID_RUNTIME(
                  src_it != view_map.end() && dst_it != view_map.end(),
                  "AllGatherOp src/dst_in not found in view_map");

              const auto& src = src_it->second;
              const auto& dst = dst_it->second;

              if (i == 0) {
                pending.send_count = src.count;
                pending.dtype = src.dtype;
              }

              parts.insert(src.participant);
              parts.insert(dst.participant);

              pending.participants.push_back(AllGatherParticipant{
                  .participant = dst.participant,
                  .send_ref = src.buffer_ref,
                  .send_offset_bytes = src.offset_bytes,
                  .recv_ref = dst.buffer_ref,
                  .recv_offset_bytes = dst.offset_bytes,
              });

              // dst_out inherits dst view info
              view_map.try_emplace(concrete.dst_outs[i], dst_it->second);
            }

            pending_all_gathers.push_back(std::move(pending));

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

  // Set up N-way communicators for AllGather collectives
  for (const auto& ag : pending_all_gathers) {
    Participants ag_parts;
    for (const auto& p : ag.participants) {
      ag_parts.insert(p.participant);
    }

    if (!comm_cache_.contains(ag_parts)) {
      ncclUniqueId nccl_id;
      ncclGetUniqueId(&nccl_id);
      auto comm_id = CommId::From(nccl_id);

      DeviceRank rank = 0;
      std::unordered_map<Participant, DeviceRank> ranks;
      for (const auto& p : ag_parts) {
        ranks[p] = rank++;
      }
      comm_cache_[ag_parts] = CommCacheEntry{.id = comm_id, .ranks = ranks};

      for (const auto& p : ag_parts) {
        programs[p].emplace_back(llc::InitComm(comm_id, ranks));
      }
    }
  }

  // === Step 3: Data-dependency-driven emission ===
  //
  // Emit copies in CIR program order.  Use SyncPoint/Wait for fine-grained
  // dependency tracking: SyncPoint is emitted after a write, Wait before a
  // read that conflicts with a prior write.  Only writes that are actually
  // read get a SyncPoint (each one forces ncclGroupEnd in the executor).

  struct WriteRegion {
    ref::BufferRef buffer;
    std::size_t offset;
    std::size_t size;
    std::uint32_t sync_id;
  };

  std::unordered_map<Participant, std::vector<WriteRegion>> pending_writes;
  std::uint32_t next_sync_id = 0;

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

  // Emit Wait for each pending write that overlaps the read region.
  auto resolve_conflicts = [&](const Participant& part,
                               const ref::BufferRef& buf, std::size_t offset,
                               std::size_t size) {
    auto it = pending_writes.find(part);
    if (it == pending_writes.end()) return;
    std::unordered_set<std::uint32_t> emitted;
    for (const auto& w : it->second) {
      if (w.buffer == buf && w.offset < offset + size &&
          offset < w.offset + w.size) {
        if (emitted.insert(w.sync_id).second) {
          flush_copy_batches_for(part);
          programs[part].emplace_back(llc::Wait(w.sync_id));
        }
      }
    }
  };

  // Precompute which buffers are ever read from.
  // This lets us skip SyncPoint emission for writes to buffers that are
  // never read, without scanning pending_copies each time.
  struct BufferRefHash {
    std::size_t operator()(const ref::BufferRef& b) const {
      return hash_value(b);
    }
  };
  std::unordered_set<ref::BufferRef, BufferRefHash> read_buffers;
  for (const auto& c : pending_copies) {
    read_buffers.insert(c.src_ref);
  }

  // Record a write.  Only emit SyncPoint if any copy reads from this buffer.
  auto record_write = [&](const Participant& part, const ref::BufferRef& buf,
                          std::size_t offset, std::size_t size) {
    if (!read_buffers.contains(buf)) return;
    auto sync_id = next_sync_id++;
    programs[part].emplace_back(llc::SyncPoint(sync_id));
    pending_writes[part].push_back({buf, offset, size, sync_id});
  };

  for (const auto& c : pending_copies) {
    auto op_size = c.count * torch::elementSize(c.dtype);
    if (c.src_part == c.dst_part) {
      resolve_conflicts(c.src_part, c.src_ref, c.src_offset_bytes, op_size);
      copy_batches[c.src_part].emplace_back(c.src_ref, c.src_offset_bytes,
                                            c.dst_ref, c.dst_offset_bytes,
                                            c.count, c.dtype);
      flush_copy_batches_for(c.dst_part);
      record_write(c.dst_part, c.dst_ref, c.dst_offset_bytes, op_size);
    } else {
      resolve_conflicts(c.src_part, c.src_ref, c.src_offset_bytes, op_size);

      Participants pair_parts;
      pair_parts.insert(c.src_part);
      pair_parts.insert(c.dst_part);
      const auto& pair_entry = comm_cache_.at(pair_parts);

      flush_copy_batches_for(c.src_part);
      flush_copy_batches_for(c.dst_part);

      programs[c.src_part].emplace_back(
          llc::Send(pair_entry.id, c.src_ref, c.src_offset_bytes, c.count,
                    c.dtype, pair_entry.ranks.at(c.dst_part)));
      programs[c.dst_part].emplace_back(
          llc::Receive(pair_entry.id, c.dst_ref, c.dst_offset_bytes, c.count,
                       c.dtype, pair_entry.ranks.at(c.src_part)));

      record_write(c.dst_part, c.dst_ref, c.dst_offset_bytes, op_size);
    }
  }

  flush_all_copy_batches();

  // === Step 4: Emit AllGather instructions ===
  for (const auto& ag : pending_all_gathers) {
    Participants ag_parts;
    for (const auto& p : ag.participants) {
      ag_parts.insert(p.participant);
    }
    const auto& cache_entry = comm_cache_.at(ag_parts);
    auto num_ranks = static_cast<DeviceRank>(ag.participants.size());

    for (const auto& p : ag.participants) {
      auto send_size = ag.send_count * torch::elementSize(ag.dtype);
      auto recv_size = send_size * num_ranks;
      resolve_conflicts(p.participant, p.send_ref, p.send_offset_bytes,
                        send_size);
      resolve_conflicts(p.participant, p.recv_ref, p.recv_offset_bytes,
                        recv_size);

      programs[p.participant].emplace_back(llc::AllGather(
          cache_entry.id, p.send_ref, p.send_offset_bytes, p.recv_ref,
          p.recv_offset_bytes, ag.send_count, ag.dtype, num_ranks));

      record_write(p.participant, p.recv_ref, p.recv_offset_bytes, recv_size);
    }
  }

  return plan;
}

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
