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
#include "planner/passes/PassContext.h"
//==============================================================================
namespace setu::planner {
//==============================================================================
Planner::Planner(targets::BackendPtr backend,
                 passes::PassManagerPtr pass_manager)
    : backend_(std::move(backend)), pass_manager_(std::move(pass_manager)) {
  ASSERT_VALID_POINTER_ARGUMENT(backend_);
}
//==============================================================================
CompileResult Planner::Compile(const CopySpec& spec, MetaStore& metastore,
                               const HintStore& hints,
                               CopyOperationId copy_op_id) {
  setu::telemetry::CompilationMetrics cm;
  cm.copy_op_id = copy_op_id;

  auto t_total = std::chrono::high_resolution_clock::now();

  // Stage 1: CopySpec -> CIR
  auto t0 = std::chrono::high_resolution_clock::now();
  auto cir = planner::passes::CopySpecToCIR::Run(spec, metastore);
  double stage1_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - t0)
                         .count();
  cm.pass_timings.push_back({"CopySpecToCIR", stage1_ms});

  // Stage 2: Optimization passes (timed individually)
  passes::PassContext ctx{.hints = hints, .register_sets = register_sets_};
  auto [optimized_cir, pass_timings] =
      pass_manager_->RunTimed(std::move(cir), ctx);
  cm.pass_timings.insert(cm.pass_timings.end(), pass_timings.begin(),
                         pass_timings.end());

  // Stage 3: Backend lowering
  t0 = std::chrono::high_resolution_clock::now();
  Plan plan = backend_->Run(optimized_cir);
  double stage3_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - t0)
                         .count();
  cm.pass_timings.push_back({"Backend", stage3_ms});

  cm.total_compile_time_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - t_total)
          .count();
  cm.num_participants = static_cast<std::uint32_t>(plan.participants.size());
  for (const auto& [p, prog] : plan.program) {
    cm.participant_instruction_counts.emplace_back(
        p.ToString(), static_cast<std::uint32_t>(prog.size()));
  }

  return {std::move(plan), std::move(cm)};
}
//==============================================================================
void Planner::AddBackendRegisterSets(
    const std::unordered_map<ir::cir::Device, RegisterSet>& register_sets) {
  backend_->AddRegisterSets(register_sets);
  for (const auto& [device, reg_set] : register_sets) {
    register_sets_.insert_or_assign(device, reg_set);
  }
}
//==============================================================================
}  // namespace setu::planner
//==============================================================================
