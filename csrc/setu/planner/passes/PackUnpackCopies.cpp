#include "planner/passes/PackUnpackCopies.h"

namespace setu::planner::passes {

using setu::planner::Participant;

using OpIndex = std::size_t;
using CopyOpIndices = std::vector<OpIndex>;
using DevicePair = std::pair<Participant, Participant>;

cir::Program PackUnpackCopies::Run(cir::Program program,
                                   const HintStore& /*hints*/) {
  // Phase 1: Group cross-device CopyOps by (src_device, dst_device) pair.
  // Within each device pair, copies are further sub-grouped by dtype.
  std::map<DevicePair, std::map<torch::Dtype, CopyOpIndices>> groups;

  for (OpIndex i = 0; i < program.NumOperations(); ++i) {
    const auto& op = program.Operations()[i];
    if (op.Type() != cir::OpType::kCopy) {
      continue;
    }

    const auto& copy = std::get<cir::CopyOp>(op.op);
    auto src_info = program.GetValueInfo(copy.src);
    auto dst_info = program.GetValueInfo(copy.dst_in);

    // Only group cross-device copies
    if (src_info.device == dst_info.device) {
      continue;
    }

    groups[{src_info.device, dst_info.device}][src_info.dtype].push_back(i);
  }

  // Collect all grouped copy op indices so we can skip them during cloning.
  std::set<OpIndex> grouped_op_indices;
  for (const auto& [device_pair, dtype_map] : groups) {
    for (const auto& [dtype, op_indices] : dtype_map) {
      grouped_op_indices.insert(op_indices.begin(), op_indices.end());
    }
  }

  // Phase 2: Rewrite the program.
  // CIR is a dataflow DAG -- operation order only matters for def-before-use.
  // Clone all non-grouped ops first (this maps all ViewOps that define the
  // sources and destinations used by the copies), then emit pack/copy/unpack
  // sequences for each group.
  auto rw = cir::ProgramRewriter(program);

  for (OpIndex i = 0; i < program.NumOperations(); ++i) {
    if (!grouped_op_indices.contains(i)) {
      rw.CloneOp(i);
    }
  }

  // Emit pack -> copy -> unpack for each group
  for (const auto& [device_pair, dtype_map] : groups) {
    const auto& [src_device, dst_device] = device_pair;

    for (const auto& [dtype, op_indices] : dtype_map) {
      if (op_indices.size() < 2) {
        // Singleton -- emit as plain copy
        const auto& copy =
            std::get<cir::CopyOp>(program.Operations()[op_indices[0]].op);
        auto new_dst_out =
            rw.Target().EmitCopy(rw.Lookup(copy.src), rw.Lookup(copy.dst_in));
        rw.MapValue(copy.dst_out, new_dst_out);
        continue;
      }

      // Collect sources and destinations, compute total size
      std::vector<cir::Value> mapped_srcs;
      std::vector<cir::Value> mapped_dst_ins;
      std::vector<cir::Value> old_dst_outs;
      std::size_t total_size = 0;

      for (auto idx : op_indices) {
        const auto& copy = std::get<cir::CopyOp>(program.Operations()[idx].op);
        mapped_srcs.push_back(rw.Lookup(copy.src));
        mapped_dst_ins.push_back(rw.Lookup(copy.dst_in));
        old_dst_outs.push_back(copy.dst_out);
        total_size += program.GetValueInfo(copy.src).size_elements;
      }

      // 1. Pack all sources into a temp on the source device
      auto src_tmp = rw.Target().EmitAllocTmp(src_device, total_size, dtype);
      auto src_tmp_packed = rw.Target().EmitPack(mapped_srcs, src_tmp);

      // 2. Single cross-device copy
      auto dst_tmp = rw.Target().EmitAllocTmp(dst_device, total_size, dtype);
      auto dst_tmp_copied = rw.Target().EmitCopy(src_tmp_packed, dst_tmp);

      // 3. Unpack into original destinations
      auto unpack_results =
          rw.Target().EmitUnpack(dst_tmp_copied, mapped_dst_ins);

      for (std::size_t j = 0; j < old_dst_outs.size(); ++j) {
        rw.MapValue(old_dst_outs[j], unpack_results[j]);
      }
    }
  }

  return rw.Finish();
}

}  // namespace setu::planner::passes
