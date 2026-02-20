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
#include "planner/hints/HintStore.h"
//==============================================================================
namespace setu::planner::hints {
//==============================================================================
void HintStore::AddHint(CompilerHint hint) {
  std::lock_guard<std::mutex> lock(mutex_);
  hints_.emplace_back(std::move(hint));
}
//==============================================================================
HintStore HintStore::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  HintStore snapshot;
  snapshot.hints_ = hints_;
  return snapshot;
}
//==============================================================================
void HintStore::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  hints_.clear();
}
//==============================================================================
std::size_t HintStore::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return hints_.size();
}
//==============================================================================
}  // namespace setu::planner::hints
//==============================================================================
