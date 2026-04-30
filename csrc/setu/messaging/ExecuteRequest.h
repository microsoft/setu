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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
//==============================================================================
#include "commons/Types.h"
#include "commons/utils/Serialization.h"
#include "messaging/BaseRequest.h"
#include "planner/Planner.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::planner::Plan;
//==============================================================================

struct ExecuteRequest : public BaseRequest {
  /// @brief Constructs a request with auto-generated request ID.
  ExecuteRequest(CopyOperationId copy_op_id_param, Plan node_plan_param)
      : BaseRequest(),
        copy_op_id(copy_op_id_param),
        node_plan(std::move(node_plan_param)) {}

  /// @brief Constructs a request with explicit request ID (for
  /// deserialization).
  ExecuteRequest(RequestId request_id_param, CopyOperationId copy_op_id_param,
                 Plan node_plan_param)
      : BaseRequest(request_id_param),
        copy_op_id(copy_op_id_param),
        node_plan(std::move(node_plan_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "ExecuteRequest(request_id={}, copy_op_id={}, node_plan={})",
        request_id, boost::uuids::to_string(copy_op_id), node_plan.ToString());
  }

  void Serialize(BinaryBuffer& buffer) const;

  static ExecuteRequest Deserialize(const BinaryRange& range);

  // NOTE: removed `const` from these members so the implicit move-ctor is
  // not deleted; otherwise boost::concurrent::sync_queue::pull() falls back
  // to copy via std::move_if_noexcept and pays a deep-copy of the Plan on
  // every dequeue (~1.7 ms / message in the Sarvam-30B-TP=2 workload).
  CopyOperationId copy_op_id;
  Plan node_plan;
};
using ExecuteRequestPtr = std::shared_ptr<ExecuteRequest>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
