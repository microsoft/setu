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

/// %dst_out = copy(%src, %dst_in)
///
/// Copies data from src to dst, producing a new SSA version of dst.
/// The lifetime of dst_in terminates at this point; it must be referred
/// to as dst_out henceforth. src and dst_in must have the same
/// size_elements.
struct CopyOp {
  Value dst_out;  ///< New version of destination after the copy
  Value src;      ///< Source value (read)
  Value dst_in;   ///< Destination value before copy (consumed)

  [[nodiscard]] std::string ToString() const {
    return std::format("{} = copy({}, {})", dst_out.ToString(), src.ToString(),
                       dst_in.ToString());
  }
};

//==============================================================================
}  // namespace setu::cir
//==============================================================================
