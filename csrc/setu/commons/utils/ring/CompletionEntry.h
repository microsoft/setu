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
#include "commons/Types.h"
#include "commons/enums/Enums.h"
#include "commons/utils/ring/SPSCRing.h"
//==============================================================================
namespace setu::commons::utils::ring {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::enums::ErrorCode;
//==============================================================================

/// @brief Entry written by the NodeAgent into the completion ring when a copy
/// operation finishes.
struct CompletionEntry {
  CopyOperationId copy_op_id;  // 16 bytes (boost::uuid)
  ErrorCode error_code;        // 4 bytes
  std::uint32_t _pad;          // 4 bytes — pad to 24 bytes
};

static_assert(std::is_trivially_copyable_v<CompletionEntry>,
              "CompletionEntry must be trivially copyable for SPSC ring");

using CompletionRingProducer = SPSCRingProducer<CompletionEntry>;
using CompletionRingConsumer = SPSCRingConsumer<CompletionEntry>;

//==============================================================================
}  // namespace setu::commons::utils::ring
//==============================================================================
