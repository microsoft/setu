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
#include "planner/hints/Hint.h"
//==============================================================================
namespace setu::planner::hints {
//==============================================================================

class HintStore {
 public:
  HintStore() = default;

  HintStore(HintStore&& other) noexcept : hints_(std::move(other.hints_)) {}

  HintStore& operator=(HintStore&& other) noexcept {
    if (this != &other) {
      hints_ = std::move(other.hints_);
    }
    return *this;
  }

  HintStore(const HintStore&) = delete;
  HintStore& operator=(const HintStore&) = delete;

  void AddHint(CompilerHint hint);

  [[nodiscard]] HintStore Snapshot() const;

  template <typename T>
  [[nodiscard]] std::vector<std::reference_wrapper<const T>> GetHints() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::reference_wrapper<const T>> result;
    for (const auto& hint : hints_) {
      if (auto* ptr = std::get_if<T>(&hint)) {
        result.emplace_back(*ptr);
      }
    }
    return result;
  }

  void Clear();

  [[nodiscard]] std::size_t Size() const;

 private:
  mutable std::mutex mutex_;
  std::vector<CompilerHint> hints_;
};

//==============================================================================
}  // namespace setu::planner::hints
//==============================================================================
