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
#include <gtest/gtest.h>
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/ir/cir/Analysis.h"
#include "planner/ir/cir/Program.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Linearity;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::cir::Value;
//==============================================================================
namespace {
//==============================================================================

Device MakeDevice(std::int16_t gpu_index = 0) {
  auto node_id = boost::uuids::nil_uuid();
  return Device(node_id, setu::commons::datatypes::Device(torch::Device(
                             torch::kCUDA, static_cast<int8_t>(gpu_index))));
}

setu::planner::ir::ref::ShardRef MakeShardRef() {
  return setu::planner::ir::ref::ShardRef(boost::uuids::nil_uuid());
}

//==============================================================================
// Valid programs pass linearity check
//==============================================================================

TEST(CIRLinearityTest, ValidProgram_CopyChain_Passes) {
  // %0 = view(dev, shard, [0,64], f16)
  // %1 = alloc_tmp(dev, 64, f16)
  // %2 = copy(%0, %1)          -- consumes %1, produces %2
  // %3 = alloc_tmp(dev, 64, f16)
  // %4 = copy(%2, %3)          -- consumes %3, uses %2 (not consumed)
  Program program;
  auto dev = MakeDevice();
  auto shard = MakeShardRef();

  auto v0 = program.EmitView(dev, shard, Slice{0, 64}, torch::kFloat16);
  auto v1 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto v2 = program.EmitCopy(v0, v1);
  auto v3 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitCopy(v2, v3);

  EXPECT_NO_THROW(Linearity::Check(program));
}

TEST(CIRLinearityTest, ReadOnlyOperandReused_Passes) {
  // %0 is read (not consumed) by two separate copies -- this is fine.
  // %0 = view(dev, shard, [0,64], f16)
  // %1 = alloc_tmp(dev, 64, f16)
  // %2 = copy(%0, %1)          -- reads %0, consumes %1
  // %3 = alloc_tmp(dev, 64, f16)
  // %4 = copy(%0, %3)          -- reads %0 again, consumes %3
  Program program;
  auto dev = MakeDevice();
  auto shard = MakeShardRef();

  auto v0 = program.EmitView(dev, shard, Slice{0, 64}, torch::kFloat16);
  auto v1 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitCopy(v0, v1);
  auto v3 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitCopy(v0, v3);

  EXPECT_NO_THROW(Linearity::Check(program));
}

TEST(CIRLinearityTest, EmptyProgram_Passes) {
  Program program;
  EXPECT_NO_THROW(Linearity::Check(program));
}

//==============================================================================
// Use-after-consume violations
//==============================================================================

TEST(CIRLinearityTest, UseAfterConsume_Copy_Throws) {
  // %0 = view(dev, shard, [0,64], f16)
  // %1 = alloc_tmp(dev, 64, f16)
  // %2 = copy(%0, %1)          -- consumes %1
  // %3 = alloc_tmp(dev, 64, f16)
  // %4 = copy(%1, %3)          -- ERROR: %1 was already consumed
  Program program;
  auto dev = MakeDevice();
  auto shard = MakeShardRef();

  auto v0 = program.EmitView(dev, shard, Slice{0, 64}, torch::kFloat16);
  auto v1 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitCopy(v0, v1);
  auto v3 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitCopy(v1, v3);

  EXPECT_THROW(Linearity::Check(program), std::runtime_error);
}

TEST(CIRLinearityTest, UseAfterConsume_ConsumedAsReadOperand_Throws) {
  // %0 = alloc_tmp(dev, 64, f16)
  // %1 = alloc_tmp(dev, 64, f16)
  // %2 = copy(%0, %1)          -- consumes %1
  // %3 = alloc_tmp(dev, 64, f16)
  // %4 = copy(%1, %3)          -- ERROR: %1 consumed, even as read operand
  Program program;
  auto dev = MakeDevice();

  auto v0 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto v1 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitCopy(v0, v1);
  auto v3 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitCopy(v1, v3);

  EXPECT_THROW(Linearity::Check(program), std::runtime_error);
}

TEST(CIRLinearityTest, UseAfterConsume_Unpack_Throws) {
  // %0 = alloc_tmp(dev, 128, f16)
  // %1 = alloc_tmp(dev, 64, f16)
  // %2 = alloc_tmp(dev, 64, f16)
  // (%3, %4) = unpack(%0, (%1, %2))  -- consumes %1 and %2
  // %5 = alloc_tmp(dev, 64, f16)
  // %6 = copy(%1, %5)                -- ERROR: %1 was consumed by unpack
  Program program;
  auto dev = MakeDevice();

  auto v0 = program.EmitAllocTmp(dev, 128, torch::kFloat16);
  auto v1 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  auto v2 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitUnpack(v0, {v1, v2});
  auto v5 = program.EmitAllocTmp(dev, 64, torch::kFloat16);
  (void)program.EmitCopy(v1, v5);

  EXPECT_THROW(Linearity::Check(program), std::runtime_error);
}

//==============================================================================
}  // namespace
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
