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
#include "planner/ir/cir/Operation.h"
#include "planner/ir/llc/Instruction.h"
#include "planner/ir/ref/BufferRef.h"
#include "planner/targets/DataDependence.h"
#include "planner/targets/NcclEmitInternal.h"
//==============================================================================
namespace setu::planner::targets {
//==============================================================================

namespace llc = setu::planner::ir::llc;
namespace ref = setu::planner::ir::ref;

namespace {

//==============================================================================
// Internal types for the DAG path.
//==============================================================================

/// Key for per-(node, emission_participant) SyncPoint id allocation.
/// A SyncPoint after a node's instruction on σ signals completion of
/// everything the node did on σ (reads and writes). This keying
/// therefore covers RAW, WAW, and WAR uniformly: every edge
/// `pred -> succ` syncs by having succ Wait on sync ids for each of
/// pred's emission participants.
struct SyncKey {
  std::uint32_t node;
  Participant part;
  bool operator==(const SyncKey& o) const {
    return node == o.node && part == o.part;
  }
};
struct SyncKeyHash {
  std::size_t operator()(const SyncKey& k) const noexcept {
    return std::hash<std::uint32_t>{}(k.node) ^
           (std::hash<Participant>{}(k.part) << 1);
  }
};
using SyncIdMap = std::unordered_map<SyncKey, std::uint32_t, SyncKeyHash>;

/// Precomputed emission participants per DAG node; reused by
/// AllocateSyncIds and EmitOneFrontier.
using EmissionParticipantsMap =
    std::unordered_map<std::uint32_t, std::set<Participant>>;

/// Everything emission needs to drive the frontier walk, precomputed
/// once up front so emission itself contains no derivation logic.
struct DagAnalysis {
  DataDependence dag;
  Participants participants;
  EmissionParticipantsMap emission_participants;
  SyncIdMap sync_ids;
};

//==============================================================================
// Analysis helpers.
//==============================================================================

/// Participants whose LLC program gets at least one instruction when
/// this node is lowered.
std::set<Participant> EmittingParticipantsOf(
    const DataDependenceNode& node /*[in]*/,
    const cir::Program& program /*[in]*/,
    const setu::planner::passes::PassContext& ctx /*[in]*/) {
  std::set<Participant> out;
  const auto& cir_op = program.Operations()[node.op_idx];
  std::visit(
      [&](const auto& concrete) {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, cir::CopyOp> ||
                      std::is_same_v<T, cir::PackOp> ||
                      std::is_same_v<T, cir::UnpackOp>) {
          for (std::size_t i = 0; i < node.reads.size(); ++i) {
            const auto& sp = node.reads[i].participant;
            const auto& dp = node.writes[i].participant;
            if (sp == dp) {
              out.insert(dp);  // llc::Copy in dp's program
            } else if (ctx.HasP2PAccess(sp, dp)) {
              out.insert(dp);  // llc::Pull in dp's program
            } else {
              out.insert(sp);  // llc::Send in sp's program
              out.insert(dp);  // llc::Receive in dp's program
            }
          }
        } else if constexpr (std::is_same_v<T, cir::AllGatherOp>) {
          // One llc::AllGather per participant.
          for (const auto& r : node.reads) out.insert(r.participant);
        }
      },
      cir_op.op);
  return out;
}

Participants CollectParticipants(const DataDependence& dag /*[in]*/) {
  Participants parts;
  for (const auto& node : dag.nodes) {
    for (const auto& p : node.participants) parts.insert(p);
  }
  return parts;
}

/// For every DAG node, precompute its emission participants.
EmissionParticipantsMap BuildEmissionParticipants(
    const DataDependence& dag /*[in]*/,
    const cir::Program& program /*[in]*/,
    const setu::planner::passes::PassContext& ctx /*[in]*/) {
  EmissionParticipantsMap out;
  for (std::uint32_t n = 0; n < dag.nodes.size(); ++n) {
    out[n] = EmittingParticipantsOf(dag.nodes[n], program, ctx);
  }
  return out;
}

/// Allocate one sync id per (node, emission_participant) for every
/// node that has at least one successor. Nodes whose results no one
/// reads get no SyncPoints.
SyncIdMap AllocateSyncIds(
    const DataDependence& dag /*[in]*/,
    const EmissionParticipantsMap& emission_parts /*[in]*/) {
  SyncIdMap out;
  std::uint32_t next_id = 0;
  for (std::uint32_t n = 0; n < dag.nodes.size(); ++n) {
    if (dag.succs[n].empty()) continue;
    for (const auto& part : emission_parts.at(n)) {
      out[{n, part}] = next_id++;
    }
  }
  return out;
}

//==============================================================================
// Top-level analysis: CIR -> DagAnalysis.
//==============================================================================

DagAnalysis AnalyzeDag(
    const cir::Program& program /*[in]*/,
    const std::optional<cir::RegisterAllocation>& reg_alloc /*[in]*/,
    const setu::planner::passes::PassContext& ctx /*[in]*/) {
  DagAnalysis out;
  out.dag = BuildDataDependence(program, reg_alloc);
  out.participants = CollectParticipants(out.dag);
  ASSERT_VALID_RUNTIME(!out.participants.empty(),
                       "DataDependence produced no participants");
  out.emission_participants =
      BuildEmissionParticipants(out.dag, program, ctx);
  out.sync_ids = AllocateSyncIds(out.dag, out.emission_participants);
  return out;
}

//==============================================================================
// Comm setup for the DAG path: discover required pair/group comms by
// walking DAG nodes, then initialize them via `comm_cache`.
//==============================================================================

void EnsureCommsForDag(
    const DagAnalysis& analysis /*[in]*/,
    const cir::Program& program /*[in]*/,
    const setu::planner::passes::PassContext& ctx /*[in]*/,
    const UniqueIdGenerator& unique_id_gen /*[in]*/,
    CommCache& comm_cache /*[inout]*/,
    std::unordered_map<Participant, std::vector<llc::Instruction>>&
        programs /*[inout]*/) {
  std::set<Participants> pair_parts;
  std::vector<Participants> allgather_groups;
  for (const auto& node : analysis.dag.nodes) {
    const auto& cir_op = program.Operations()[node.op_idx];
    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;
          if constexpr (std::is_same_v<T, cir::CopyOp> ||
                        std::is_same_v<T, cir::PackOp> ||
                        std::is_same_v<T, cir::UnpackOp>) {
            for (std::size_t i = 0; i < node.reads.size(); ++i) {
              const auto& sp = node.reads[i].participant;
              const auto& dp = node.writes[i].participant;
              if (sp == dp) continue;
              if (ctx.HasP2PAccess(sp, dp)) continue;
              Participants pp;
              pp.insert(sp);
              pp.insert(dp);
              pair_parts.insert(pp);
            }
          } else if constexpr (std::is_same_v<T, cir::AllGatherOp>) {
            Participants group;
            for (const auto& r : node.reads) group.insert(r.participant);
            allgather_groups.push_back(std::move(group));
          }
        },
        cir_op.op);
  }

  auto init_comm = [&](const Participants& pp) {
    if (comm_cache.contains(pp)) return;
    auto comm_id = llc::CommId::From(unique_id_gen());
    DeviceRank rank = 0;
    std::unordered_map<Participant, DeviceRank> ranks;
    for (const auto& p : pp) ranks[p] = rank++;
    comm_cache[pp] = CommCacheEntry{.id = comm_id, .ranks = ranks};
    for (const auto& p : pp) {
      programs[p].emplace_back(llc::InitComm(comm_id, ranks));
    }
  };
  for (const auto& pp : pair_parts) init_comm(pp);
  for (const auto& pp : allgather_groups) init_comm(pp);
}

//==============================================================================
// Per-frontier emission.
//==============================================================================

/// Produce this frontier's Waits, memcpy batches, extra collectives,
/// and SyncPoints, then flush them into the per-participant programs
/// in canonical order.
void EmitOneFrontier(
    const std::vector<std::uint32_t>& frontier /*[in]*/,
    const DagAnalysis& analysis /*[in]*/,
    const cir::Program& program /*[in]*/,
    const setu::planner::passes::PassContext& ctx /*[in]*/,
    const CommCache& comm_cache /*[in]*/,
    std::unordered_map<Participant, std::vector<llc::Instruction>>&
        programs /*[inout]*/) {
  const auto& dag = analysis.dag;

  std::unordered_map<Participant, std::vector<std::uint32_t>> waits;
  std::unordered_map<Participant, std::unordered_set<std::uint32_t>> wait_seen;
  std::unordered_map<Participant, std::vector<BatchEntry>> batches;
  std::unordered_map<Participant, std::vector<llc::Instruction>> extras;
  std::unordered_map<Participant, std::vector<std::uint32_t>> syncs;
  std::unordered_map<Participant, std::unordered_set<std::uint32_t>> sync_seen;

  auto add_wait = [&](const Participant& part, std::uint32_t sid) {
    if (wait_seen[part].insert(sid).second) waits[part].push_back(sid);
  };
  auto add_sync = [&](const Participant& part, std::uint32_t sid) {
    if (sync_seen[part].insert(sid).second) syncs[part].push_back(sid);
  };

  for (std::uint32_t n : frontier) {
    const auto& node_s = dag.nodes[n];

    // (a) Waits: for every pred, wait on each of pred's sync ids on
    //     each of this node's emission participants. One sync id per
    //     (pred, pred_emission_participant), so this may wait on
    //     several ids per pred (for example both sides of an NCCL
    //     Send/Recv, or every participant of an AllGather). Correct
    //     for RAW, WAW, and WAR because a SyncPoint after a node's
    //     instruction on σ signals completion of everything that
    //     instruction did on σ — both reads and writes.
    const auto& emit_parts = analysis.emission_participants.at(n);
    for (std::uint32_t p : dag.preds[n]) {
      for (const auto& pred_part : analysis.emission_participants.at(p)) {
        auto sid = analysis.sync_ids.at({p, pred_part});
        for (const auto& part : emit_parts) add_wait(part, sid);
      }
    }

    // (b) Body: derive LLC from node.reads + node.writes + CIR op type.
    const auto& cir_op = program.Operations()[node_s.op_idx];
    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;
          if constexpr (std::is_same_v<T, cir::CopyOp> ||
                        std::is_same_v<T, cir::PackOp> ||
                        std::is_same_v<T, cir::UnpackOp>) {
            ASSERT_VALID_RUNTIME(
                node_s.reads.size() == node_s.writes.size(),
                "DAG node {} (memcpy op) has {} reads vs {} writes", n,
                node_s.reads.size(), node_s.writes.size());
            for (std::size_t i = 0; i < node_s.reads.size(); ++i) {
              const auto& r = node_s.reads[i];
              const auto& w = node_s.writes[i];
              auto elt_size =
                  static_cast<std::size_t>(torch::elementSize(r.dtype));
              auto count = (r.end_bytes - r.start_bytes) / elt_size;
              const auto& src_part = r.participant;
              const auto& dst_part = w.participant;
              if (src_part == dst_part) {
                batches[dst_part].push_back(BatchEntry{
                    .kind = BatchKind::kCopy,
                    .src_ref = r.buffer_ref,
                    .src_offset_bytes = r.start_bytes,
                    .dst_ref = w.buffer_ref,
                    .dst_offset_bytes = w.start_bytes,
                    .count = count,
                    .dtype = r.dtype,
                    .src_device = {},
                });
              } else if (ctx.HasP2PAccess(src_part, dst_part)) {
                batches[dst_part].push_back(BatchEntry{
                    .kind = BatchKind::kPull,
                    .src_ref = r.buffer_ref,
                    .src_offset_bytes = r.start_bytes,
                    .dst_ref = w.buffer_ref,
                    .dst_offset_bytes = w.start_bytes,
                    .count = count,
                    .dtype = r.dtype,
                    .src_device = src_part.device.GetDeviceId(),
                });
              } else {
                Participants pair_parts;
                pair_parts.insert(src_part);
                pair_parts.insert(dst_part);
                const auto& ce = comm_cache.at(pair_parts);
                extras[src_part].emplace_back(
                    llc::Send(ce.id, r.buffer_ref, r.start_bytes, count,
                              r.dtype, ce.ranks.at(dst_part)));
                extras[dst_part].emplace_back(
                    llc::Receive(ce.id, w.buffer_ref, w.start_bytes, count,
                                 r.dtype, ce.ranks.at(src_part)));
              }
            }
          } else if constexpr (std::is_same_v<T, cir::AllGatherOp>) {
            Participants ag_parts;
            for (const auto& r : node_s.reads) ag_parts.insert(r.participant);
            const auto& ce = comm_cache.at(ag_parts);
            auto num_ranks = static_cast<DeviceRank>(ag_parts.size());
            ASSERT_VALID_RUNTIME(
                node_s.reads.size() == node_s.writes.size(),
                "AllGather DAG node {} reads/writes size mismatch", n);
            for (std::size_t i = 0; i < node_s.reads.size(); ++i) {
              const auto& r = node_s.reads[i];
              const auto& w = node_s.writes[i];
              ASSERT_VALID_RUNTIME(
                  r.participant == w.participant,
                  "AllGather DAG node {} mismatched read/write participant",
                  n);
              auto elt_size =
                  static_cast<std::size_t>(torch::elementSize(r.dtype));
              auto send_count = (r.end_bytes - r.start_bytes) / elt_size;
              extras[r.participant].emplace_back(llc::AllGather(
                  ce.id, r.buffer_ref, r.start_bytes, w.buffer_ref,
                  w.start_bytes, send_count, r.dtype, num_ranks));
            }
          } else {
            RAISE_RUNTIME_ERROR("DAG node {} wraps unsupported CIR op type",
                                n);
          }
        },
        cir_op.op);

    // (c) SyncPoints: one per emission participant of n, but only if
    //     n has any successor (AllocateSyncIds skipped it otherwise).
    //     Flagging each emission side individually lets successors
    //     wait on the specific side they depend on.
    for (const auto& part : emit_parts) {
      auto sit = analysis.sync_ids.find({n, part});
      if (sit != analysis.sync_ids.end()) add_sync(part, sit->second);
    }
  }

  // Flush per participant, in canonical order:
  //   Waits -> batched Copy -> batched Pull
  //        -> Send/Recv/AllGather -> SyncPoints.
  std::set<Participant> touched;
  for (const auto& [p, _] : waits) touched.insert(p);
  for (const auto& [p, _] : batches) touched.insert(p);
  for (const auto& [p, _] : extras) touched.insert(p);
  for (const auto& [p, _] : syncs) touched.insert(p);

  for (const auto& p : touched) {
    auto& prog = programs[p];
    if (auto it = waits.find(p); it != waits.end()) {
      for (auto sid : it->second) prog.emplace_back(llc::Wait(sid));
    }
    if (auto it = batches.find(p); it != batches.end()) {
      std::vector<llc::CopyEntry> copy_run;
      std::vector<llc::PullEntry> pull_run;
      for (const auto& e : it->second) {
        if (e.kind == BatchKind::kCopy) {
          copy_run.emplace_back(e.src_ref, e.src_offset_bytes, e.dst_ref,
                                e.dst_offset_bytes, e.count, e.dtype);
        } else {
          pull_run.emplace_back(e.src_ref, e.src_offset_bytes, e.dst_ref,
                                e.dst_offset_bytes, e.count, e.dtype,
                                e.src_device);
        }
      }
      if (!copy_run.empty())
        prog.emplace_back(llc::Copy(std::move(copy_run)));
      if (!pull_run.empty())
        prog.emplace_back(llc::Pull(std::move(pull_run)));
    }
    if (auto it = extras.find(p); it != extras.end()) {
      for (auto& instr : it->second) prog.emplace_back(std::move(instr));
    }
    if (auto it = syncs.find(p); it != syncs.end()) {
      for (auto sid : it->second) prog.emplace_back(llc::SyncPoint(sid));
    }
  }
}

//==============================================================================
// Frontier walk driver (Kahn's algorithm).
//==============================================================================

void EmitFrontiers(
    const DagAnalysis& analysis /*[in]*/,
    const cir::Program& program /*[in]*/,
    const setu::planner::passes::PassContext& ctx /*[in]*/,
    const CommCache& comm_cache /*[in]*/,
    std::unordered_map<Participant, std::vector<llc::Instruction>>&
        programs /*[inout]*/) {
  const auto& dag = analysis.dag;
  std::vector<std::uint32_t> indeg(dag.nodes.size());
  std::vector<std::uint32_t> frontier;
  for (std::uint32_t n = 0; n < dag.nodes.size(); ++n) {
    indeg[n] = static_cast<std::uint32_t>(dag.preds[n].size());
    if (indeg[n] == 0) frontier.push_back(n);
  }

  while (!frontier.empty()) {
    EmitOneFrontier(frontier, analysis, program, ctx, comm_cache, programs);
    std::vector<std::uint32_t> next_frontier;
    for (std::uint32_t n : frontier) {
      for (std::uint32_t s : dag.succs[n]) {
        if (--indeg[s] == 0) next_frontier.push_back(s);
      }
    }
    frontier = std::move(next_frontier);
  }
}

}  // namespace

//==============================================================================

void NCCL::EmitWithDataDependence(
    const cir::Program& program,
    const std::optional<cir::RegisterAllocation>& reg_alloc,
    const setu::planner::passes::PassContext& ctx, Plan& plan) {
  auto analysis = AnalyzeDag(program, reg_alloc, ctx);
  plan.participants = analysis.participants;
  EnsureCommsForDag(analysis, program, ctx, unique_id_gen_, comm_cache_,
                    plan.program);
  EmitFrontiers(analysis, program, ctx, comm_cache_, plan.program);
}

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
