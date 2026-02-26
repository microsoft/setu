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
#include "planner/passes/PassManager.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================
void PassManager::AddPass(PassPtr pass) {
  ASSERT_VALID_POINTER_ARGUMENT(pass);
  passes_.emplace_back(std::move(pass));
}
//==============================================================================
cir::Program PassManager::Run(cir::Program program,
                              const HintStore& hints) const {
  for (const auto& pass : passes_) {
    program = pass->Run(std::move(program), hints);
    LOG_DEBUG("After pass '{}': {}", pass->Name(), program.Dump());
  }
  return program;
}
//==============================================================================
std::size_t PassManager::NumPasses() const { return passes_.size(); }
//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
