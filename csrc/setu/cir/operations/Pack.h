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
#include "cir/Value.h"
//==============================================================================
namespace setu::cir {
//==============================================================================

/// %dst_out = pack((%src0, ..., %srcN), %dst_in)
///
/// Concatenates multiple source buffers into a contiguous destination.
/// The total size of all sources must equal the destination size. Sources are
/// packed in order.
struct PackOp {
  Value dst_out;            ///< New version of destination after the pack
  std::vector<Value> srcs;  ///< Source values (read, concatenated in order)
  Value dst_in;             ///< Destination value before pack (consumed)

  [[nodiscard]] std::string ToString() const {
    std::string srcs_str;
    for (std::size_t i = 0; i < srcs.size(); ++i) {
      if (i > 0) srcs_str += ", ";
      srcs_str += srcs[i].ToString();
    }
    return std::format("{} = pack(({}), {})", dst_out.ToString(), srcs_str,
                       dst_in.ToString());
  }
};

//==============================================================================
}  // namespace setu::cir
//==============================================================================
