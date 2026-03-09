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
#include "planner/ir/cir/Value.h"
//==============================================================================
namespace setu::planner::ir::cir {
//==============================================================================

/// (%dst0_out, ..., %dstN_out) = all_gather(%src0, %dst0_in, ..., %srcN,
///                                           %dstN_in)
///
/// Collective operation: N participants each contribute a chunk (src_i), and
/// the full concatenation of all chunks is written to each participant's
/// destination buffer (dst_i).
///
/// Invariants:
///   - srcs.size() == dst_ins.size() == dst_outs.size() == N
///   - All src_i have the same size_elements
///   - Each dst_i_in.size_elements == N * src_i.size_elements
///   - dst_i values are consumed (ownership transfer)
struct AllGatherOp {
  std::vector<Value> dst_outs;  ///< New versions of destinations after gather
  std::vector<Value> srcs;      ///< Source chunks (one per participant)
  std::vector<Value> dst_ins;  ///< Destination buffers before gather (consumed)

  [[nodiscard]] std::string ToString() const {
    std::string dst_outs_str;
    std::string srcs_str;
    std::string dst_ins_str;
    for (std::size_t i = 0; i < dst_outs.size(); ++i) {
      if (i > 0) {
        dst_outs_str += ", ";
        srcs_str += ", ";
        dst_ins_str += ", ";
      }
      dst_outs_str += dst_outs[i].ToString();
      srcs_str += srcs[i].ToString();
      dst_ins_str += dst_ins[i].ToString();
    }
    return std::format("({}) = all_gather(srcs=[{}], dst_ins=[{}])",
                       dst_outs_str, srcs_str, dst_ins_str);
  }
};

//==============================================================================
}  // namespace setu::planner::ir::cir
//==============================================================================
