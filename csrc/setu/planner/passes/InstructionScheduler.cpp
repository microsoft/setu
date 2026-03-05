#include "planner/passes/InstructionScheduler.h"

#include "commons/Logging.h"

namespace setu::planner::passes {

/// Instruction Scheduling for CIR operations.
///
/// At the moment, this pass is used to alleviate register pressure from
/// AllocTmpOps.
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
/// schedule with bounded register pressure:
///
///   AllocTmp(c0), Copy(s0, c0), Copy(c0_out, d0),
///   AllocTmp(c1), Copy(s1, c1), Copy(c1_out, d1), ...
cir::Program InstructionScheduler::Run(cir::Program program,
                                       const HintStore& /*hints*/) {
  const auto num_ops = program.NumOperations();
  if (num_ops <= 1) {
    return program;
  }

  // Build `in_degree` and `successors` map to facilitate toposort.
  // An operation depends on all of its inputs.
  std::vector<std::uint32_t> in_degree(num_ops, 0);
  std::vector<std::vector<std::uint32_t>> successors(num_ops);
  for (std::uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    std::unordered_set<std::uint32_t> preds;
    for (const auto& used : op.Uses()) {
      auto def_op = program.GetValueInfo(used).def_op_index;
      if (def_op != op_idx && preds.insert(def_op).second) {
        ++in_degree[op_idx];
        successors[def_op].push_back(op_idx);
      }
    }
  }

  // Score each operation.
  //
  // AllocTmpOps get -1 (they increase register pressure by starting new
  // work).  All other ops get 0 (they make progress on existing work,
  // moving values toward their last use).  This single distinction is
  // enough to produce the interleaved schedule described above.
  std::vector<std::int32_t> score(num_ops, 0);
  for (std::uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    if (program.Operations()[op_idx].Type() == cir::OpType::kAllocTmp) {
      score[op_idx] = -1;
    }
  }

  // Perform priority-based toposort.
  using Entry = std::pair<std::int32_t, std::int32_t>;
  std::priority_queue<Entry> ready;
  for (std::uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    if (in_degree[op_idx] == 0) {
      ready.emplace(score[op_idx], -static_cast<std::int32_t>(op_idx));
    }
  }

  std::vector<std::size_t> schedule;
  schedule.reserve(num_ops);

  while (!ready.empty()) {
    auto [_, neg_idx] = ready.top();
    ready.pop();
    auto op_idx = static_cast<std::uint32_t>(-neg_idx);

    schedule.push_back(op_idx);

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

}  // namespace setu::planner::passes
