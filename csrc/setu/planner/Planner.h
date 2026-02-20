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
#include "commons/datatypes/CopySpec.h"
#include "metastore/MetaStore.h"
#include "planner/Plan.h"
#include "planner/hints/HintStore.h"
#include "planner/passes/PassManager.h"
#include "planner/targets/backend.h"
//==============================================================================
namespace setu::planner {
//==============================================================================

using setu::commons::datatypes::CopySpec;
using setu::metastore::MetaStore;
using setu::planner::hints::HintStore;
using setu::planner::ir::llc::Program;

class Planner {
 public:
  Planner(targets::BackendPtr backend, passes::PassManagerPtr pass_manager);
  [[nodiscard]] Plan Compile(const CopySpec& spec, MetaStore& metastore,
                             const HintStore& hints);

 private:
  targets::BackendPtr backend_;
  passes::PassManagerPtr pass_manager_;
};

using PlannerPtr = std::shared_ptr<Planner>;

//==============================================================================
}  // namespace setu::planner
//==============================================================================
