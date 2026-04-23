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
#include "planner/ir/llc/Instruction.h"
#include "planner/targets/NcclEmitInternal.h"
//==============================================================================
namespace setu::planner::targets {
//==============================================================================

namespace llc = setu::planner::ir::llc;

namespace {

/// After emission, each SyncPoint needs its `wait_count` set to the
/// number of Waits that reference its id.
void BackpatchSyncPointWaitCounts(
    std::unordered_map<Participant, std::vector<llc::Instruction>>&
        programs /*[inout]*/) {
  std::unordered_map<std::uint32_t, std::uint32_t> wait_counts;
  for (const auto& [part, prog] : programs) {
    for (const auto& instr : prog) {
      if (auto* w = std::get_if<llc::Wait>(&instr.instr)) {
        wait_counts[w->id]++;
      }
    }
  }
  for (auto& [part, prog] : programs) {
    for (auto& instr : prog) {
      if (auto* sp = std::get_if<llc::SyncPoint>(&instr.instr)) {
        auto it = wait_counts.find(sp->id);
        sp->wait_count = (it != wait_counts.end()) ? it->second : 0;
      }
    }
  }
}

/// Real ncclUniqueId generator used in production.
ncclUniqueId RealUniqueId() {
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  return id;
}

}  // namespace

//==============================================================================

NCCL::NCCL() : unique_id_gen_(&RealUniqueId) {}

NCCL::NCCL(UniqueIdGenerator unique_id_gen)
    : unique_id_gen_(std::move(unique_id_gen)) {
  ASSERT_VALID_ARGUMENTS(static_cast<bool>(unique_id_gen_),
                         "NCCL: unique_id_gen must be callable");
}

Plan NCCL::Run(const cir::Program& program,
               const setu::planner::passes::PassContext& ctx) {
  Plan plan;

  bool has_alloc_tmp = std::ranges::any_of(
      program.Operations(),
      [](const auto& op) { return op.Type() == cir::OpType::kAllocTmp; });
  std::optional<cir::RegisterAllocation> reg_alloc;
  if (has_alloc_tmp) {
    auto liveness = cir::LivenessInfo::Build(program);
    reg_alloc =
        cir::RegisterAllocation::Build(program, liveness, ctx.register_sets);
  }

  EmitWithDataDependence(program, reg_alloc, ctx, plan);

  BackpatchSyncPointWaitCounts(plan.program);
  return plan;
}

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
