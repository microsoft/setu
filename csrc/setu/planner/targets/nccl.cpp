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

          } else {
            RAISE_RUNTIME_ERROR(
                "NCCL backend: unsupported CIR operation (only view, "
                "alloc_tmp, slice, copy, and consume are supported)");
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

  // === Step 3: Staged emission — emit Barrier between depth stages ===

  auto copy_depth = cir::CopyDepthAnalysis::Build(program);

  // Sort pending copies by depth so we can iterate once and insert barriers
  // at stage boundaries. Order within a stage is irrelevant — all copies at
  // the same depth are independent.
  std::ranges::sort(pending_copies, [&](const PendingCopy& a,
                                        const PendingCopy& b) {
    return copy_depth.depth[a.cir_op_index] < copy_depth.depth[b.cir_op_index];
  });

  std::uint32_t prev_stage = 0;
  for (const auto& c : pending_copies) {
    auto stage = copy_depth.depth[c.cir_op_index].value();

    if (stage != prev_stage) {
      for (const auto& part : parts) {
        programs[part].emplace_back(llc::Barrier());
      }
      prev_stage = stage;
    }

    if (c.src_part == c.dst_part) {
      programs[c.src_part].emplace_back(llc::Copy(c.src_ref, c.src_offset_bytes,
                                                  c.dst_ref, c.dst_offset_bytes,
                                                  c.count, c.dtype));
    } else {
      programs[c.src_part].emplace_back(llc::Send(c.src_ref, c.src_offset_bytes,
                                                  c.count, c.dtype,
                                                  entry.ranks.at(c.dst_part)));
      programs[c.dst_part].emplace_back(
          llc::Receive(c.dst_ref, c.dst_offset_bytes, c.count, c.dtype,
                       entry.ranks.at(c.src_part)));
    }
  }

  return plan;
}

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
