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
#include "planner/ir/cir/Analysis.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================

// ========================= Linearity =======================================

void Linearity::Check(const Program& program) {
  std::unordered_set<Value> consumed;
  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    for (const auto& used : op.Uses()) {
      ASSERT_VALID_RUNTIME(
          consumed.find(used) == consumed.end(),
          "Linearity violation at op [{}] {}: value {} was already consumed",
          op_idx, op.ToString(), used.ToString());
    }
    for (const auto& val : op.ConsumedOperands()) {
      consumed.insert(val);
    }
  }
}

// ========================= DefUseChains ====================================

DefUseChains DefUseChains::Build(const Program& program) {
  DefUseChains chains;
  chains.uses.resize(program.NumValues());

  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    for (const auto& used_value : op.Uses()) {
      chains.uses[used_value.id].push_back(op_idx);
    }
  }

  return chains;
}

// ========================= LivenessInfo ====================================

LivenessInfo LivenessInfo::Build(const Program& program) {
  LivenessInfo info;
  info.ranges.resize(program.NumValues());

  // Initialize: each value is defined at its def_op_index, last_use = first_def
  for (std::uint32_t v = 0; v < program.NumValues(); ++v) {
    auto def_idx = program.GetValueInfo(Value{v}).def_op_index;
    info.ranges[v] = LiveRange{.first_def = def_idx, .last_use = def_idx};
  }

  // Extend last_use based on actual uses.
  // Because we don't have control flow, no fixpoint iteration needed :)
  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    for (const auto& used_value : op.Uses()) {
      auto& range = info.ranges[used_value.id];
      range.last_use = std::max(range.last_use, op_idx);
    }
  }

  return info;
}

std::vector<Value> LivenessInfo::LiveAt(std::uint32_t op_index) const {
  std::vector<Value> result;
  for (std::uint32_t v = 0; v < ranges.size(); ++v) {
    const auto& range = ranges[v];
    if (range.first_def <= op_index && op_index <= range.last_use) {
      result.push_back(Value{v});
    }
  }
  return result;
}

// ========================= RegisterAllocation ==============================

RegisterAllocation RegisterAllocation::Build(
    const Program& program, const LivenessInfo& liveness,
    const std::unordered_map<Device, setu::planner::RegisterSet>&
        register_sets) {
  // NOTE: We currently assume all registers within a RegisterSet have the
  // same size (uniform pools).  The allocator treats registers as fungible
  // slots and does not match AllocTmpOp sizes to individual register sizes.
  RegisterAllocation result;
  result.allocation.resize(program.NumValues());

  // Per-device: priority queue of (available_after, register_index), min-heap
  using SlotQueue =
      std::priority_queue<std::pair<std::uint32_t, std::uint32_t>,
                          std::vector<std::pair<std::uint32_t, std::uint32_t>>,
                          std::greater<>>;

  std::unordered_map<Device, SlotQueue> free_slots;
  std::unordered_map<Device, std::uint32_t> next_slot_index;

  // Initialize free slot pools from RegisterSets
  for (const auto& [device, reg_set] : register_sets) {
    auto num_regs = reg_set.NumRegisters();
    auto& queue = free_slots[device];
    for (std::uint32_t i = 0; i < num_regs; ++i) {
      queue.emplace(0, i);  // all slots available from op 0
    }
    next_slot_index[device] = num_regs;
  }

  // Collect AllocTmpOp values sorted by their def op index (already in order)
  std::vector<Value> tmp_values;
  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    if (op.Type() == OpType::kAllocTmp) {
      for (const auto& def : op.Defs()) {
        tmp_values.push_back(def);
      }
    }
  }

  // Linear scan: allocate physical registers for each AllocTmpOp value
  for (const auto& val : tmp_values) {
    const auto& val_info = program.GetValueInfo(val);
    const auto& live_range = liveness.ranges[val.id];
    const auto& device = val_info.device;

    auto set_it = register_sets.find(device);
    ASSERT_VALID_RUNTIME(set_it != register_sets.end(),
                         "No register set configured for device {}",
                         device.ToString());

    auto& queue = free_slots[device];

    // Free any slots whose live ranges have ended before this def
    // (they're at the top of the min-heap if available)
    // Note: slots become reusable after their last_use, so check
    // available_after <= live_range.first_def
    // No action needed -- the heap naturally gives us the earliest-available

    // Find a free slot: one whose available_after <= first_def
    std::vector<std::pair<std::uint32_t, std::uint32_t>> deferred;
    std::optional<std::uint32_t> assigned_slot;

    while (!queue.empty()) {
      auto [avail_after, slot_idx] = queue.top();
      if (avail_after <= live_range.first_def) {
        queue.pop();
        assigned_slot = slot_idx;
        break;
      }
      // This slot isn't free yet; no earlier one will be either (min-heap)
      break;
    }

    ASSERT_VALID_RUNTIME(assigned_slot.has_value(),
                         "Register allocation failed for {} on device {}: "
                         "pool exhausted ({} registers)",
                         val.ToString(), device.ToString(),
                         set_it->second.NumRegisters());

    result.allocation[val.id].emplace(
        PhysicalRegister{.device = device, .register_index = *assigned_slot});

    // Return slot to pool after this value's last use
    queue.emplace(live_range.last_use + 1, *assigned_slot);
  }

  return result;
}

// ========================= CopyDepthAnalysis ================================

CopyDepthAnalysis CopyDepthAnalysis::Build(const Program& program) {
  CopyDepthAnalysis result;
  result.depth.resize(program.NumOperations());
  result.max_depth = 0;

  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    if (op.Type() != OpType::kCopy) {
      continue;
    }

    const auto& copy_op = std::get<CopyOp>(op.op);

    // Trace the src operand back through SliceOp/ConsumeOp chains to find
    // the root defining operation.
    Value current = copy_op.src;
    while (true) {
      const auto& def_op = program.GetDefiningOp(current);
      if (def_op.Type() == OpType::kSlice) {
        current = std::get<SliceOp>(def_op.op).src;
      } else if (def_op.Type() == OpType::kConsume) {
        current = std::get<ConsumeOp>(def_op.op).src;
      } else {
        break;
      }
    }

    // current now points to a value defined by the root op.
    // Check if the root is a CopyOp (meaning this is a relay chain).
    const auto& root_op = program.GetDefiningOp(current);
    const auto& root_info = program.GetValueInfo(current);
    auto root_op_idx = root_info.def_op_index;

    if (root_op.Type() == OpType::kCopy) {
      ASSERT_VALID_RUNTIME(
          result.depth[root_op_idx].has_value(),
          "CopyOp at index {} depends on CopyOp at index {} which has no "
          "depth (topological order violation?)",
          op_idx, root_op_idx);
      result.depth[op_idx] = result.depth[root_op_idx].value() + 1;
    } else {
      result.depth[op_idx] = 0;
    }

    result.max_depth = std::max(result.max_depth, result.depth[op_idx].value());
  }

  return result;
}

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================
