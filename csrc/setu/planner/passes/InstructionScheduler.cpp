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

/// Instruction Scheduling for CIR operations.
///
/// After RegisterTiling, the program contains flat sequences of AllocTmpOps
/// followed by their associated copies:
///
///   AllocTmp(c0), AllocTmp(c1), ...,
///   Copy(s0, c0), Copy(s1, c1), ...,     // writes into tmps
///   Copy(c0_out, d0), Copy(c1_out, d1), ...  // reads out of tmps
///
/// All AllocTmpOps are independent and thus simultaneously ready for
/// scheduling.  Without reordering the register allocator would see all
/// tmps live at once and exhaust its pool.
///
/// The scheduler performs a priority-based topological sort. AllocTmpOps
/// (which create new live registers) get score -1; every other op gets
/// score 0.  Among ready ops the highest score wins, with original program
/// order as tie-break.  This causes the scheduler to drain each tmp's
/// copy chain before allocating the next one, producing an interleaved
/// schedule with bounded register pressure.
///
/// When register_sets are provided, a pressure guard additionally defers
/// AllocTmpOps that would exceed the device's register pool.  Registers
/// are tracked via alias-chain use-counts (refcounting): when all uses of
/// a root AllocTmpOp's alias chain are exhausted, the register is freed
/// and deferred AllocTmpOps for that device are re-enqueued.
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

  // Build `in_degree` and `successors` map to facilitate toposort.
  // Also compute use_count per alias root (for pressure tracking).
  std::vector<std::uint32_t> in_degree(num_ops, 0);
  std::vector<std::vector<std::uint32_t>> successors(num_ops);
  std::unordered_map<cir::Value, std::uint32_t> use_count;

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

  // Score each operation.
  std::vector<std::int32_t> score(num_ops, 0);
  for (std::uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    if (program.Operations()[op_idx].Type() == cir::OpType::kAllocTmp) {
      score[op_idx] = -1;
    }
  }

  // Priority-based toposort with optional pressure guard.
  using Entry = std::pair<std::int32_t, std::int32_t>;
  std::priority_queue<Entry> ready;
  for (std::uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    if (in_degree[op_idx] == 0) {
      ready.emplace(score[op_idx], -static_cast<std::int32_t>(op_idx));
    }
  }

  // Pressure guard state.
  std::unordered_map<cir::Device, std::uint32_t> live_registers;
  std::unordered_map<cir::Value, std::uint32_t> remaining_uses;
  std::vector<Entry> deferred;

  std::vector<std::size_t> schedule;
  schedule.reserve(num_ops);

  while (!ready.empty() || !deferred.empty()) {
    // Pop from ready queue, deferring AllocTmpOps that would exceed pressure.
    std::optional<std::uint32_t> chosen;
    while (!ready.empty()) {
      auto [s, neg_idx] = ready.top();
      auto op_idx = static_cast<std::uint32_t>(-neg_idx);

      if (has_pressure_guard &&
          program.Operations()[op_idx].Type() == cir::OpType::kAllocTmp) {
        const auto& alloc_op =
            std::get<cir::AllocTmpOp>(program.Operations()[op_idx].op);
        auto set_it = register_sets_.find(alloc_op.device);
        if (set_it != register_sets_.end() &&
            live_registers[alloc_op.device] >= set_it->second.NumRegisters()) {
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

    // If this is an AllocTmpOp, start tracking its register.
    if (has_pressure_guard && op.Type() == cir::OpType::kAllocTmp) {
      const auto& alloc_op = std::get<cir::AllocTmpOp>(op.op);
      live_registers[alloc_op.device]++;
      remaining_uses[alloc_op.out] = use_count[alloc_op.out];
    }

    // Decrement use counts for alias roots and free registers.
    if (has_pressure_guard) {
      std::unordered_set<cir::Device> freed_devices;
      for (const auto& used : op.Uses()) {
        if (alias.root[used.id].has_value()) {
          auto root_val = alias.root[used.id].value();
          auto it = remaining_uses.find(root_val);
          if (it != remaining_uses.end()) {
            it->second--;
            if (it->second == 0) {
              auto root_device = program.GetValueInfo(root_val).device;
              live_registers[root_device]--;
              remaining_uses.erase(it);
              freed_devices.insert(root_device);
            }
          }
        }
      }

      // Re-enqueue deferred AllocTmpOps whose devices now have capacity.
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

    // Update successors.
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
