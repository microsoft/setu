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
#include "planner/Participant.h"
#include "planner/topo/Topology.h"
//==============================================================================
namespace setu::planner::hints {
//==============================================================================
using setu::planner::Participant;
using setu::planner::topo::Path;
//==============================================================================

struct RoutingHint {
  Participant src;
  Participant dst;
  Path path;

  RoutingHint(Participant src_param, Participant dst_param, Path path_param)
      : src(std::move(src_param)),
        dst(std::move(dst_param)),
        path(std::move(path_param)) {}

  [[nodiscard]] std::string ToString() const {
    return std::format("RoutingHint(src={}, dst={}, path={})", src, dst, path);
  }
};

using CompilerHint = std::variant<RoutingHint>;

//==============================================================================
}  // namespace setu::planner::hints
//==============================================================================
