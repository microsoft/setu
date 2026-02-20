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
//==============================================================================
#include "planner/RegisterSet.h"
#include "planner/ir/cir/Program.h"
#include "planner/ir/cir/Value.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================
/// Linearity check: verifies that consumed operands (dst_in in copy/pack,
/// dst_ins in unpack) are never used after their consuming operation.
/// A consumed value is dead -- subsequent operations must use the new SSA
/// version (dst_out) instead.
struct Linearity {
  /// Checks linearity of the program.
  /// Asserts on the first violation with a descriptive error message.
  static void Check(const Program& program /*[in]*/);
};
//==============================================================================

/// Def-use chains: for each Value, which operations use it.
/// Defs are already stored in ValueInfo::def_op_index.
struct DefUseChains {
  /// Indexed by Value::id -> list of operation indices that use this value
  std::vector<std::vector<std::uint32_t>> uses;

  /// Build def-use chains from a Program
  [[nodiscard]] static DefUseChains Build(const Program& program /*[in]*/);
};

//==============================================================================

/// Liveness analysis: for each Value, the operation index range over which
/// it is live [first_def, last_use].
struct LivenessInfo {
  struct LiveRange {
    std::uint32_t first_def;  ///< Op index where value is defined
    std::uint32_t last_use;   ///< Op index of last use (inclusive)

    [[nodiscard]] std::string ToString() const {
      return std::format("[{}, {}]", first_def, last_use);
    }
  };

  /// Indexed by Value::id
  std::vector<LiveRange> ranges;

  /// Returns the set of values live at a given operation index
  [[nodiscard]] std::vector<Value> LiveAt(
      std::uint32_t op_index /*[in]*/) const;

  [[nodiscard]] static LivenessInfo Build(const Program& program /*[in]*/);
};

//==============================================================================

/// Register allocation result: maps virtual registers (from AllocTmpOp)
/// to physical buffer pool slots. Uses linear scan over liveness intervals
/// to reuse physical slots when live ranges do not overlap.
/// No spilling -- asserts if allocation is impossible (pool exhausted).
struct RegisterAllocation {
  /// A physical register: a (device, index) pair identifying a slot
  /// in the device's pre-allocated temporary buffer pool.
  struct PhysicalRegister {
    Device device;
    std::uint32_t register_index;

    [[nodiscard]] std::string ToString() const {
      return std::format("PhysReg({}, #{})", device.ToString(), register_index);
    }
  };

  /// Indexed by Value::id. Only populated for AllocTmpOp-defined values.
  std::vector<std::optional<PhysicalRegister>> allocation;

  /// Build register allocation using linear scan on liveness info.
  /// Only allocates registers for AllocTmpOp-defined values.
  /// Asserts if any device's pool is exhausted.
  ///
  /// @param program The CIR program
  /// @param liveness Pre-computed liveness info
  /// @param register_sets Per-device register set specifying available slots
  [[nodiscard]] static RegisterAllocation Build(
      const Program& program /*[in]*/, const LivenessInfo& liveness /*[in]*/,
      const std::unordered_map<Device, setu::planner::RegisterSet>&
          register_sets /*[in]*/);
};

//==============================================================================

/// Copy-depth analysis: for each CopyOp, the number of CopyOp ancestors
/// along the src chain.  Used to stage multi-hop relay copies so that
/// NCCL sends at depth N only execute after depth N-1 receives complete.
struct CopyDepthAnalysis {
  /// Indexed by CIR operation index. Only populated for CopyOp operations.
  /// Depth 0 = src has no CopyOp ancestor (reads directly from View/AllocTmp).
  /// Depth N = src traces through Slice/Consume to a CopyOp at depth N-1.
  std::vector<std::optional<std::uint32_t>> depth;

  /// Maximum depth across all CopyOps (0 if no copies, or all are depth 0).
  std::uint32_t max_depth;

  [[nodiscard]] static CopyDepthAnalysis Build(const Program& program);
};

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================
