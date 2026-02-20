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
#pragma once
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "commons/ClassTraits.h"
#include "planner/ir/cir/Operation.h"
#include "planner/ir/cir/Slice.h"
#include "planner/ir/cir/Value.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================

/// A CIR Program is both the container for the SSA dataflow graph and the
/// builder that emits operations into it.
///
/// Operations are stored in a flat vector (no control flow, no basic blocks).
/// The DAG structure is implicit in the def-use edges. Emission order is a
/// valid topological order by construction (operands must be defined before
/// use).
///
/// The Program is append-only: operations cannot be removed or reordered.
/// Optimization passes use ProgramRewriter to walk an existing Program and
/// build a new one with transformations applied.
class Program : public setu::commons::NonCopyable {
 public:
  Program() = default;

  // ==================== Builder API ====================

  /// %out = view(device, handle, slice, dtype)
  [[nodiscard]] Value EmitView(
      const Device& device /*[in]*/,
      const setu::planner::ir::ref::ShardRef& handle /*[in]*/,
      Slice slice /*[in]*/, torch::Dtype dtype /*[in]*/);

  /// %out = alloc_tmp(device, size_elements, dtype)
  [[nodiscard]] Value EmitAllocTmp(const Device& device /*[in]*/,
                                   std::size_t size_elements /*[in]*/,
                                   torch::Dtype dtype /*[in]*/);

  /// %out = slice(%src, slice)
  /// Requires: slice.offset + slice.size <= src.size_elements
  [[nodiscard]] Value EmitSlice(Value src /*[in]*/, Slice slice /*[in]*/);

  /// %dst_out = copy(%src, %dst_in)
  /// Requires: src.size_elements == dst_in.size_elements
  [[nodiscard]] Value EmitCopy(Value src /*[in]*/, Value dst_in /*[in]*/);

  /// %out = consume(%src)
  /// Consumes src, returning a new SSA value for the underlying buffer.
  [[nodiscard]] Value EmitConsume(Value src /*[in]*/);

  /// %dst_out = pack(srcs, %dst_in)
  /// Requires: sum(src_i.size_elements) == dst_in.size_elements
  [[nodiscard]] Value EmitPack(std::vector<Value> srcs /*[in]*/,
                               Value dst_in /*[in]*/);

  /// (%dst0_out, ..., %dstN_out) = unpack(%src, (%dst0_in, ..., %dstN_in))
  /// Requires: src.size_elements == sum(dst_in_i.size_elements)
  [[nodiscard]] std::vector<Value> EmitUnpack(
      Value src /*[in]*/, std::vector<Value> dst_ins /*[in]*/);

  // ==================== Query API ====================

  [[nodiscard]] const std::vector<Operation>& Operations() const {
    return ops_;
  }

  [[nodiscard]] std::size_t NumOperations() const { return ops_.size(); }

  [[nodiscard]] std::size_t NumValues() const { return value_info_.size(); }

  [[nodiscard]] const ValueInfo& GetValueInfo(Value v) const;

  [[nodiscard]] const Operation& GetDefiningOp(Value v) const;

  // ==================== Debug ====================

  /// One-line summary: "Program(ops=N, values=M)"
  [[nodiscard]] std::string ToString() const;

  /// Multi-line dump of all operations
  [[nodiscard]] std::string Dump() const;

 private:
  [[nodiscard]] Value AllocateValue(const Device& device,
                                    std::size_t size_elements,
                                    torch::Dtype dtype,
                                    std::uint32_t def_op_index);

  std::vector<Operation> ops_;
  std::vector<ValueInfo> value_info_;  ///< Indexed by Value::id
  std::uint32_t next_value_id_{0};
};

//==============================================================================

/// Maps Values from an old Program to Values in a new Program.
using ValueMap = std::unordered_map<Value, Value>;

//==============================================================================

/// Utility for rebuild-based optimization passes.
///
/// A pass walks the old Program's operations in order. For each operation,
/// it can either re-emit it as-is (with remapped values) via CloneOp, or
/// emit a custom replacement sequence directly into Target() and MapValue
/// the results.
///
/// Usage:
///   ProgramRewriter rewriter(old_program);
///   for (std::size_t i = 0; i < old_program.NumOperations(); ++i) {
///     const auto& op = old_program.Operations()[i];
///     if (ShouldRewrite(op)) {
///       // 1:N rewrite: emit multiple ops into Target()
///       auto tmp = rewriter.Target().EmitAllocTmp(...);
///       auto tmp_out = rewriter.Target().EmitCopy(rewriter.Lookup(src), tmp);
///       auto result = rewriter.Target().EmitCopy(tmp_out,
///       rewriter.Lookup(dst)); rewriter.MapValue(old_result, result);
///     } else {
///       rewriter.CloneOp(i);  // 1:1 pass-through with value remapping
///     }
///   }
///   Program new_program = rewriter.Finish();
class ProgramRewriter {
 public:
  explicit ProgramRewriter(const Program& source /*[in]*/);

  /// Returns the target Program being built. Passes emit replacement
  /// operations directly into this program.
  [[nodiscard]] Program& Target() { return target_; }

  /// Looks up the new Value corresponding to an old Value.
  /// Asserts if the old value has not been mapped yet.
  [[nodiscard]] Value Lookup(Value old_value /*[in]*/) const;

  /// Explicitly maps an old Value to a new Value. Used when a pass
  /// emits custom replacement ops and needs to record the correspondence.
  void MapValue(Value old_value /*[in]*/, Value new_value /*[in]*/);

  /// Re-emits the operation at the given index from the source Program
  /// into the target, remapping all Value operands through the value map.
  /// Automatically maps the old result Values to the new result Values.
  void CloneOp(std::size_t op_index /*[in]*/);

  /// Finalizes the rewrite and returns the new Program.
  /// The rewriter is left in a moved-from state.
  [[nodiscard]] Program Finish();

 private:
  const Program& source_;
  Program target_;
  ValueMap value_map_;
};

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================
