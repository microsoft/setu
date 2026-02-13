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

/// (%dst0_out, ..., %dstN_out) = unpack(%src, (%dst0_in, ..., %dstN_in))
///
/// Splits a source buffer into multiple destination buffers.
/// The source size must equal the total size of all destinations. Destinations
/// are filled in order.
struct UnpackOp {
  std::vector<Value> dst_outs;  ///< New versions of destinations after unpack
  Value src;                    ///< Source value (read)
  std::vector<Value> dst_ins;   ///< Destination values before unpack (consumed)

  [[nodiscard]] std::string ToString() const {
    std::string outs_str;
    for (std::size_t i = 0; i < dst_outs.size(); ++i) {
      if (i > 0) outs_str += ", ";
      outs_str += dst_outs[i].ToString();
    }
    std::string ins_str;
    for (std::size_t i = 0; i < dst_ins.size(); ++i) {
      if (i > 0) ins_str += ", ";
      ins_str += dst_ins[i].ToString();
    }
    return std::format("({}) = unpack({}, ({}))", outs_str, src.ToString(),
                       ins_str);
  }
};

//==============================================================================
}  // namespace setu::cir
//==============================================================================
