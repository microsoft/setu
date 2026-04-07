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
  registered_names_.push_back(pass->Name());
  pass_map_[pass->Name()] = pass.get();
  passes_.emplace_back(std::move(pass));
}
//==============================================================================
cir::Program PassManager::Run(
    cir::Program program, const PassContext& ctx,
    const std::optional<std::vector<std::string>>& pass_names) const {
  const auto& names =
      pass_names.has_value() ? pass_names.value() : registered_names_;
  for (const auto& name : names) {
    program = pass_map_.at(name)->Run(std::move(program), ctx);
    LOG_DEBUG("After pass '{}': {}", name, program.Dump());
  }
  return program;
}
//==============================================================================
std::pair<cir::Program, std::vector<setu::telemetry::PassTiming>>
PassManager::RunTimed(
    cir::Program program, const PassContext& ctx,
    const std::optional<std::vector<std::string>>& pass_names) const {
  const auto& names =
      pass_names.has_value() ? pass_names.value() : registered_names_;
  std::vector<setu::telemetry::PassTiming> timings;
  timings.reserve(names.size());
  for (const auto& name : names) {
    auto t0 = std::chrono::high_resolution_clock::now();
    program = pass_map_.at(name)->Run(std::move(program), ctx);
    double elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0)
                            .count();
    timings.push_back({name, elapsed_ms});
    LOG_DEBUG("After pass '{}': {}", name, program.Dump());
  }
  return {std::move(program), std::move(timings)};
}
//==============================================================================
std::size_t PassManager::NumPasses() const { return passes_.size(); }
//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
