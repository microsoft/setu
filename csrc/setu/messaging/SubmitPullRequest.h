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
#include "commons/datatypes/CopySpec.h"
#include "commons/utils/Serialization.h"
#include "messaging/BaseRequest.h"
#include "planner/hints/Hint.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
using setu::commons::ShardId;
using setu::commons::datatypes::CopySpec;
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::planner::hints::CompilerHint;
//==============================================================================

struct SubmitPullRequest : public BaseRequest {
  /// @brief Constructs a request with auto-generated request ID.
  SubmitPullRequest(ShardId shard_id_param, CopySpec copy_spec_param,
                    std::vector<CompilerHint> hints_param = {},
                    std::uint64_t hints_fingerprint_param = 0,
                    std::uint64_t local_id_param = 0,
                    std::optional<std::vector<std::string>> pass_names_param =
                        std::nullopt)
      : BaseRequest(),
        shard_id(shard_id_param),
        copy_spec(std::move(copy_spec_param)),
        hints(std::move(hints_param)),
        hints_fingerprint(hints_fingerprint_param),
        local_id(local_id_param),
        pass_names(std::move(pass_names_param)) {}

  /// @brief Constructs a request with explicit request ID (for
  /// deserialization).
  SubmitPullRequest(RequestId request_id_param, ShardId shard_id_param,
                    CopySpec copy_spec_param,
                    std::vector<CompilerHint> hints_param,
                    std::uint64_t hints_fingerprint_param,
                    std::uint64_t local_id_param,
                    std::optional<std::vector<std::string>> pass_names_param)
      : BaseRequest(request_id_param),
        shard_id(shard_id_param),
        copy_spec(std::move(copy_spec_param)),
        hints(std::move(hints_param)),
        hints_fingerprint(hints_fingerprint_param),
        local_id(local_id_param),
        pass_names(std::move(pass_names_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format(
        "SubmitPullRequest(request_id={}, shard_id={}, copy_spec={}, "
        "num_hints={}, local_id={})",
        request_id, shard_id, copy_spec, hints.size(), local_id);
  }

  void Serialize(BinaryBuffer& buffer) const;

  static SubmitPullRequest Deserialize(const BinaryRange& range);

  const ShardId shard_id;
  const CopySpec copy_spec;
  const std::vector<CompilerHint> hints;
  const std::uint64_t hints_fingerprint;
  const std::uint64_t local_id;
  const std::optional<std::vector<std::string>> pass_names;
};
using SubmitPullRequestPtr = std::shared_ptr<SubmitPullRequest>;

//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
