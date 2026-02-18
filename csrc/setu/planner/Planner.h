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
#include "planner/targets/backend.h"
//==============================================================================
namespace setu::planner {
//==============================================================================

using setu::commons::datatypes::CopySpec;
using setu::metastore::MetaStore;
using setu::planner::ir::llc::Program;

class Planner {
 public:
  explicit Planner(std::unique_ptr<targets::Backend> backend);
  [[nodiscard]] Plan Compile(const CopySpec& spec, MetaStore& metastore);

 private:
  std::unique_ptr<targets::Backend> backend_;
};

//==============================================================================
}  // namespace setu::planner
//==============================================================================
