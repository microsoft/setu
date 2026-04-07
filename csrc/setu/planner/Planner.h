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
#include "commons/Types.h"
//==============================================================================
#include "commons/datatypes/CopySpec.h"
#include "metastore/MetaStore.h"
#include "planner/Plan.h"
#include "planner/hints/HintStore.h"
#include "planner/passes/PassContext.h"
#include "planner/passes/PassManager.h"
#include "planner/targets/backend.h"
#include "telemetry/MetricsData.h"
//==============================================================================
namespace setu::planner {
//==============================================================================

using setu::commons::CopyOperationId;
using setu::commons::NodeId;
using setu::commons::datatypes::CopySpec;
using setu::metastore::MetaStore;
using setu::planner::hints::HintStore;
using setu::planner::ir::llc::Program;
using setu::planner::passes::P2PAccessMap;

/// @brief Result of Planner::Compile, containing the Plan and compilation
/// metrics.
struct CompileResult {
  Plan plan;
  setu::telemetry::CompilationMetrics metrics;
};

class Planner {
 public:
  Planner(targets::BackendPtr backend, passes::PassManagerPtr pass_manager);
  [[nodiscard]] CompileResult Compile(
      const CopySpec& spec, MetaStore& metastore, const HintStore& hints,
      CopyOperationId copy_op_id,
      const std::optional<std::vector<std::string>>& pass_names = std::nullopt);

  /// Accumulate per-device register sets (called during NodeAgent onboarding).
  void AddBackendRegisterSets(
      const std::unordered_map<ir::cir::Device, RegisterSet>&
          register_sets /*[in]*/);

  /// Record P2P-capable device pairs for a node (called during onboarding).
  void AddP2PAccess(
      NodeId node_id /*[in]*/,
      const std::vector<passes::P2PDevicePair>& p2p_pairs /*[in]*/);

 private:
  targets::BackendPtr backend_;
  passes::PassManagerPtr pass_manager_;
  std::unordered_map<ir::cir::Device, RegisterSet> register_sets_;
  P2PAccessMap p2p_access_;
};

using PlannerPtr = std::shared_ptr<Planner>;

//==============================================================================
}  // namespace setu::planner
//==============================================================================
