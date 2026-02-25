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
    auto t0 = std::chrono::steady_clock::now();
    program = pass->Run(program, hints);
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
    LOG_INFO("PassManager: pass '{}' took {}us", pass->Name(), dt);
    LOG_DEBUG("After pass '{}': {}", pass->Name(), program.Dump());
  }
  return program;
}
//==============================================================================
std::pair<cir::Program, std::vector<setu::telemetry::PassTiming>>
PassManager::RunTimed(cir::Program program, const HintStore& hints) const {
  std::vector<setu::telemetry::PassTiming> timings;
  timings.reserve(passes_.size());
  for (const auto& pass : passes_) {
    auto t0 = std::chrono::high_resolution_clock::now();
    program = pass->Run(program, hints);
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0)
            .count();
    timings.push_back({pass->Name(), elapsed_ms});
    LOG_DEBUG("After pass '{}': {}", pass->Name(), program.Dump());
  }
  return {std::move(program), std::move(timings)};
}
//==============================================================================
std::size_t PassManager::NumPasses() const { return passes_.size(); }
//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
