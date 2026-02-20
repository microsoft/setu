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
#include "commons/ClassTraits.h"
#include "planner/passes/Pass.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================

class PassManager : public setu::commons::NonCopyable {
 public:
  PassManager() = default;
  void AddPass(PassPtr pass);
  [[nodiscard]] cir::Program Run(cir::Program program,
                                 const HintStore& hints) const;
  [[nodiscard]] std::size_t NumPasses() const;

 private:
  std::vector<PassPtr> passes_;
};

using PassManagerPtr = std::shared_ptr<PassManager>;

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
