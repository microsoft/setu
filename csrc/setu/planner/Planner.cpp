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
#include "planner/Planner.h"
//==============================================================================
#include "commons/Logging.h"
#include "planner/passes/CopySpecToCIR.h"
//==============================================================================
namespace setu::planner {
//==============================================================================
Planner::Planner(targets::BackendPtr backend,
                 passes::PassManagerPtr pass_manager)
    : backend_(std::move(backend)), pass_manager_(std::move(pass_manager)) {
  ASSERT_VALID_POINTER_ARGUMENT(backend_);
}
//==============================================================================
Plan Planner::Compile(const CopySpec& spec, MetaStore& metastore) {
  auto cir = planner::passes::CopySpecToCIR::Run(spec, metastore);
  cir = pass_manager_->Run(std::move(cir));
  return backend_->Run(cir);
}
//==============================================================================
}  // namespace setu::planner
//==============================================================================
