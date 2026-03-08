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
#include "planner/passes/InstructionScheduler.h"
//==============================================================================
#include "commons/Logging.h"
#include "planner/ir/cir/Analysis.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================

namespace {

/// Per-alias-root use count: how many times each AllocTmpOp's alias chain
/// is referenced across the program.
using UseCounts = std::unordered_map<cir::Value, std::uint32_t>;

/// Bookkeeping for register pressure simulation.  Shared between the
/// pressure feasibility check and the scheduling loop.
struct PressureTracker {
  const cir::Program& program;
  const cir::AliasChains& alias;
  const std::unordered_map<cir::Device, setu::planner::RegisterSet>&
      register_sets;
  const UseCounts& use_count;

  std::unordered_map<cir::Device, std::uint32_t> live_registers;
  std::unordered_map<cir::Value, std::uint32_t> remaining_uses;

  /// Track a newly scheduled AllocTmpOp.
  void TrackAlloc(const cir::AllocTmpOp& alloc_op) {
    live_registers[alloc_op.device]++;
    auto it = use_count.find(alloc_op.out);
    remaining_uses[alloc_op.out] = (it != use_count.end()) ? it->second : 0;
  }

  /// Returns true if scheduling an AllocTmpOp on the given device would
  /// exceed the register pool.
  [[nodiscard]] bool WouldExceedBudget(const cir::Device& device) const {
    auto set_it = register_sets.find(device);
    if (set_it == register_sets.end()) return false;
    auto it = live_registers.find(device);
    auto live = (it != live_registers.end()) ? it->second : 0u;
    return live >= set_it->second.NumRegisters();
  }

  /// Retire uses for an op and free registers whose alias chains are
  /// exhausted.  Returns the set of devices that gained capacity.
  std::unordered_set<cir::Device> RetireUses(const cir::Operation& op) {
    std::unordered_set<cir::Device> freed_devices;
    for (const auto& used : op.Uses()) {
      if (!alias.root[used.id].has_value()) continue;
      auto root_val = alias.root[used.id].value();
      auto it = remaining_uses.find(root_val);
      if (it == remaining_uses.end()) continue;
      it->second--;
      if (it->second == 0) {
        auto root_device = program.GetValueInfo(root_val).device;
        live_registers[root_device]--;
        remaining_uses.erase(it);
        freed_devices.insert(root_device);
      }
    }
    return freed_devices;
  }
};

/// Check whether the original program order fits within the register budget.
bool OriginalOrderFitsWithinBudget(
    const cir::Program& program, const cir::AliasChains& alias,
    const std::unordered_map<cir::Device, setu::planner::RegisterSet>&
        register_sets,
    const UseCounts& use_count) {
  PressureTracker tracker{program, alias, register_sets, use_count, {}, {}};

  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];

    if (op.Type() == cir::OpType::kAllocTmp) {
      const auto& alloc_op = std::get<cir::AllocTmpOp>(op.op);
      if (tracker.WouldExceedBudget(alloc_op.device)) {
        return false;
      }
      tracker.TrackAlloc(alloc_op);
    }

    tracker.RetireUses(op);
  }
  return true;
}

}  // namespace

//==============================================================================

/// Instruction Scheduling for CIR operations.
///
/// The scheduler ensures that register allocation will not fail by
/// reordering operations to keep peak register pressure within the
/// device's register pool.  If the program's original order already
/// satisfies the pressure budget, it is returned unchanged — preserving
/// any upstream ordering (e.g. wavefront order from Pipelining).
///
/// When reordering IS needed, a priority-based topological sort is used.
/// AllocTmpOps get score -1, everything else gets score 0.  A pressure
/// guard defers AllocTmpOps that would exceed the device's register pool.
cir::Program InstructionScheduler::Run(cir::Program program,
                                       const PassContext& ctx) {
  const auto& register_sets_ = ctx.register_sets;
  const auto num_ops = program.NumOperations();
  if (num_ops <= 1) {
    return program;
  }

  const bool has_pressure_guard = !register_sets_.empty();

  // Build alias chains for pressure tracking.
  cir::AliasChains alias;
  if (has_pressure_guard) {
    alias = cir::AliasChains::Build(program);
  }

  // Build dependency graph and use counts.
  std::vector<std::uint32_t> in_degree(num_ops, 0);
  std::vector<std::vector<std::uint32_t>> successors(num_ops);
  UseCounts use_count;

  for (std::uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    std::unordered_set<std::uint32_t> preds;
    for (const auto& used : op.Uses()) {
      auto def_op = program.GetValueInfo(used).def_op_index;
      if (def_op != op_idx && preds.insert(def_op).second) {
        ++in_degree[op_idx];
        successors[def_op].push_back(op_idx);
      }

      if (has_pressure_guard && alias.root[used.id].has_value()) {
        use_count[alias.root[used.id].value()]++;
      }
    }
  }

  // If the original order already fits, preserve it (e.g. wavefront from
  // Pipelining).
  if (has_pressure_guard && OriginalOrderFitsWithinBudget(
                                program, alias, register_sets_, use_count)) {
    return program;
  }

  // --- Reordering needed: priority-based toposort ---

  std::vector<std::int32_t> score(num_ops, 0);
  for (std::uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    if (program.Operations()[op_idx].Type() == cir::OpType::kAllocTmp) {
      score[op_idx] = -1;
    }
  }

  using Entry = std::pair<std::int32_t, std::int32_t>;
  std::priority_queue<Entry> ready;
  for (std::uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    if (in_degree[op_idx] == 0) {
      ready.emplace(score[op_idx], -static_cast<std::int32_t>(op_idx));
    }
  }

  PressureTracker tracker{program, alias, register_sets_, use_count, {}, {}};
  std::vector<Entry> deferred;
  std::vector<std::size_t> schedule;
  schedule.reserve(num_ops);

  while (!ready.empty() || !deferred.empty()) {
    std::optional<std::uint32_t> chosen;
    while (!ready.empty()) {
      auto [s, neg_idx] = ready.top();
      auto op_idx = static_cast<std::uint32_t>(-neg_idx);

      if (has_pressure_guard &&
          program.Operations()[op_idx].Type() == cir::OpType::kAllocTmp) {
        const auto& alloc_op =
            std::get<cir::AllocTmpOp>(program.Operations()[op_idx].op);
        if (tracker.WouldExceedBudget(alloc_op.device)) {
          ready.pop();
          deferred.push_back({s, neg_idx});
          continue;
        }
      }

      ready.pop();
      chosen = op_idx;
      break;
    }

    ASSERT_VALID_RUNTIME(
        chosen.has_value(),
        "InstructionScheduler: ready queue empty with {} deferred ops and "
        "{} scheduled out of {} total — cycle or insufficient registers",
        deferred.size(), schedule.size(), num_ops);

    auto op_idx = *chosen;
    schedule.push_back(op_idx);

    const auto& op = program.Operations()[op_idx];

    if (has_pressure_guard && op.Type() == cir::OpType::kAllocTmp) {
      tracker.TrackAlloc(std::get<cir::AllocTmpOp>(op.op));
    }

    if (has_pressure_guard) {
      auto freed_devices = tracker.RetireUses(op);

      if (!freed_devices.empty() && !deferred.empty()) {
        std::vector<Entry> still_deferred;
        for (const auto& entry : deferred) {
          auto deferred_idx = static_cast<std::uint32_t>(-entry.second);
          const auto& deferred_op =
              std::get<cir::AllocTmpOp>(program.Operations()[deferred_idx].op);
          if (freed_devices.contains(deferred_op.device)) {
            ready.push(entry);
          } else {
            still_deferred.push_back(entry);
          }
        }
        deferred = std::move(still_deferred);
      }
    }

    for (auto succ : successors[op_idx]) {
      --in_degree[succ];
      if (in_degree[succ] == 0) {
        ready.emplace(score[succ], -static_cast<std::int32_t>(succ));
      }
    }
  }

  ASSERT_VALID_RUNTIME(schedule.size() == num_ops,
                       "Scheduler produced {} ops, expected {}",
                       schedule.size(), num_ops);

  // Build the new program, if order changed.
  bool is_identity = true;
  for (std::size_t i = 0; i < num_ops; ++i) {
    if (schedule[i] != i) {
      is_identity = false;
      break;
    }
  }
  if (is_identity) {
    return program;
  }

  auto rw = cir::ProgramRewriter(program);
  for (auto op_idx : schedule) {
    rw.CloneOp(op_idx);
  }
  return rw.Finish();
}

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
