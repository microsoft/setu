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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
#include "planner/RegisterSet.h"
#include "planner/hints/HintStore.h"
#include "planner/ir/cir/Program.h"
#include "planner/ir/ref/ShardRef.h"
#include "planner/passes/PassContext.h"
#include "planner/targets/NcclEmitInternal.h"
#include "planner/targets/nccl.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
using setu::planner::RegisterSet;
using setu::planner::hints::HintStore;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Program;
using setu::planner::ir::cir::Slice;
using setu::planner::ir::ref::ShardRef;
using setu::planner::passes::P2PAccessMap;
using setu::planner::passes::P2PDevicePair;
using setu::planner::passes::PassContext;
using setu::planner::targets::NCCL;
using setu::planner::targets::UniqueIdGenerator;
//==============================================================================
namespace {

Device MakeTestDevice(std::int16_t gpu_index) {
  auto node_id = boost::uuids::nil_uuid();
  return Device(node_id, setu::commons::datatypes::Device(torch::Device(
                             torch::kCUDA, static_cast<int8_t>(gpu_index))));
}

ShardRef MakeShard(const char* id) {
  return ShardRef(boost::uuids::string_generator()(id));
}

/// Deterministic ncclUniqueId generator: embeds a monotonically
/// increasing counter in the first four bytes. Lets golden-string
/// expectations reference stable CommId hex prefixes.
UniqueIdGenerator CounterGen() {
  auto counter = std::make_shared<std::uint32_t>(0);
  return [counter]() {
    ncclUniqueId id{};
    std::uint32_t v = (*counter)++;
    std::memcpy(&id, &v, sizeof(v));
    return id;
  };
}

}  // namespace
//==============================================================================

class LoweringTest : public ::testing::Test {
 protected:
  Device dev0 = MakeTestDevice(0);
  Device dev1 = MakeTestDevice(1);
  ShardRef shard_a = MakeShard("00000000-0000-0000-0000-000000000001");
  ShardRef shard_b = MakeShard("00000000-0000-0000-0000-000000000002");
  ShardRef shard_c = MakeShard("00000000-0000-0000-0000-000000000003");
  torch::Dtype dt = torch::kFloat32;  // 4 bytes per element
  HintStore hints;
  std::unordered_map<Device, RegisterSet> empty_register_sets;
  P2PAccessMap empty_p2p;

  PassContext CtxWithP2P(const P2PAccessMap& p2p /*[in]*/) const {
    return PassContext{.hints = hints,
                       .register_sets = empty_register_sets,
                       .p2p_access = p2p};
  }

  P2PAccessMap OneWayP2P(const Device& src /*[in]*/,
                         const Device& dst /*[in]*/) const {
    P2PAccessMap p2p;
    p2p[boost::uuids::nil_uuid()].insert(P2PDevicePair{
        .src = src.device.GetDeviceId(),
        .dst = dst.device.GetDeviceId(),
    });
    return p2p;
  }
};

//==============================================================================
// A same-device Copy lowers to a single llc::Copy in the destination's
// program. No InitComm, no SyncPoint.
//==============================================================================

TEST_F(LoweringTest, SameDeviceCopy_SingleLlcCopy) {
  Program program;
  auto src = program.EmitView(dev0, shard_a, Slice{0, 16}, dt);
  auto dst = program.EmitView(dev0, shard_b, Slice{0, 16}, dt);
  (void)program.EmitCopy(src, dst);

  NCCL backend{CounterGen()};
  auto plan = backend.Run(program, CtxWithP2P(empty_p2p));

  EXPECT_EQ(plan.ToString(),
            R"(Plan
  Participants (1):
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100))

  Programs (1):
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100)) [1 instructions]:
      [0] Copy(num_entries=1)
  [0] src_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000001, tensor_name=<none>, node_id=<none>), src_offset_bytes=0, dst_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000002, tensor_name=<none>, node_id=<none>), dst_offset_bytes=0, count=16, dtype=6, src_ptr=0x0, dst_ptr=0x0

)");
}

//==============================================================================
// A cross-device Copy with P2P access lowers to a single llc::Pull
// in the destination's program.
//==============================================================================

TEST_F(LoweringTest, CrossDeviceWithP2P_SingleLlcPull) {
  Program program;
  auto src = program.EmitView(dev0, shard_a, Slice{0, 16}, dt);
  auto dst = program.EmitView(dev1, shard_b, Slice{0, 16}, dt);
  (void)program.EmitCopy(src, dst);

  NCCL backend{CounterGen()};
  auto plan = backend.Run(program, CtxWithP2P(OneWayP2P(dev0, dev1)));

  EXPECT_EQ(plan.ToString(),
            R"(Plan
  Participants (2):
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1, device_id=GPU-b0c68926-7220-fe79-3599-5a4d2a232bfb))
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100))

  Programs (1):
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1, device_id=GPU-b0c68926-7220-fe79-3599-5a4d2a232bfb)) [1 instructions]:
      [0] Pull(num_entries=1)
  [0] src_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000001, tensor_name=<none>, node_id=<none>), src_offset_bytes=0, dst_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000002, tensor_name=<none>, node_id=<none>), dst_offset_bytes=0, count=16, dtype=6, src_device=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100, src_ptr=0x0, dst_ptr=0x0

)");
}

//==============================================================================
// A cross-device Copy without P2P access lowers to an NCCL pair: an
// InitComm on both sides, a Send on the source, a Receive on the
// destination.
//==============================================================================

TEST_F(LoweringTest, CrossDeviceNoP2P_InitCommAndSendRecv) {
  Program program;
  auto src = program.EmitView(dev0, shard_a, Slice{0, 16}, dt);
  auto dst = program.EmitView(dev1, shard_b, Slice{0, 16}, dt);
  (void)program.EmitCopy(src, dst);

  NCCL backend{CounterGen()};
  auto plan = backend.Run(program, CtxWithP2P(empty_p2p));

  EXPECT_EQ(plan.ToString(),
            R"(Plan
  Participants (2):
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1, device_id=GPU-b0c68926-7220-fe79-3599-5a4d2a232bfb))
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100))

  Programs (2):
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1, device_id=GPU-b0c68926-7220-fe79-3599-5a4d2a232bfb)) [2 instructions]:
      [0] InitComm(comm_id=CommId(0000000000000000...), ranks={Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1, device_id=GPU-b0c68926-7220-fe79-3599-5a4d2a232bfb))=0, Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100))=1})
      [1] Receive(comm_id=CommId(0000000000000000...), peer_rank=1, dst_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000002, tensor_name=<none>, node_id=<none>), offset_bytes=0, count=16, dtype=6, dst_device_ptr=0x0)

    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100)) [2 instructions]:
      [0] InitComm(comm_id=CommId(0000000000000000...), ranks={Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1, device_id=GPU-b0c68926-7220-fe79-3599-5a4d2a232bfb))=0, Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100))=1})
      [1] Send(comm_id=CommId(0000000000000000...), peer_rank=0, src_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000001, tensor_name=<none>, node_id=<none>), offset_bytes=0, count=16, dtype=6, src_device_ptr=0x0)

)");
}

//==============================================================================
// Two same-device Copies in a RAW chain (shard_a -> shard_b ->
// shard_c) get a SyncPoint after the producing Copy and a Wait before
// the consuming Copy, both on dev0.
//==============================================================================

TEST_F(LoweringTest, SameDeviceChain_RawSyncPointAndWait) {
  Program program;
  auto a = program.EmitView(dev0, shard_a, Slice{0, 16}, dt);
  auto b0 = program.EmitView(dev0, shard_b, Slice{0, 16}, dt);
  (void)program.EmitCopy(a, b0);
  auto b1 = program.EmitView(dev0, shard_b, Slice{0, 16}, dt);
  auto c = program.EmitView(dev0, shard_c, Slice{0, 16}, dt);
  (void)program.EmitCopy(b1, c);

  NCCL backend{CounterGen()};
  auto plan = backend.Run(program, CtxWithP2P(empty_p2p));

  EXPECT_EQ(plan.ToString(),
            R"(Plan
  Participants (1):
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100))

  Programs (1):
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100)) [4 instructions]:
      [0] Copy(num_entries=1)
  [0] src_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000001, tensor_name=<none>, node_id=<none>), src_offset_bytes=0, dst_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000002, tensor_name=<none>, node_id=<none>), dst_offset_bytes=0, count=16, dtype=6, src_ptr=0x0, dst_ptr=0x0
      [1] SyncPoint(id=0, wait_count=1)
      [2] Wait(id=0)
      [3] Copy(num_entries=1)
  [0] src_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000002, tensor_name=<none>, node_id=<none>), src_offset_bytes=0, dst_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000003, tensor_name=<none>, node_id=<none>), dst_offset_bytes=0, count=16, dtype=6, src_ptr=0x0, dst_ptr=0x0

)");
}

//==============================================================================
// WAR regression: a P2P Pull reading shard_b on dev0 is followed by a
// same-device Copy that overwrites shard_b. Keying sync ids by
// emission participant (not write participant) ensures the overwrite
// waits on dev1's SyncPoint before starting.
//==============================================================================

TEST_F(LoweringTest, PackCopyPack_WarOnScratchIsSynced) {
  Program program;
  auto a0 = program.EmitView(dev0, shard_a, Slice{0, 16}, dt);
  auto b0 = program.EmitView(dev0, shard_b, Slice{0, 16}, dt);
  (void)program.EmitCopy(a0, b0);
  auto b1 = program.EmitView(dev0, shard_b, Slice{0, 16}, dt);
  auto c1 = program.EmitView(dev1, shard_c, Slice{0, 16}, dt);
  (void)program.EmitCopy(b1, c1);
  auto a2 = program.EmitView(dev0, shard_a, Slice{0, 16}, dt);
  auto b2 = program.EmitView(dev0, shard_b, Slice{0, 16}, dt);
  (void)program.EmitCopy(a2, b2);

  NCCL backend{CounterGen()};
  auto plan = backend.Run(program, CtxWithP2P(OneWayP2P(dev0, dev1)));

  EXPECT_EQ(plan.ToString(),
            R"(Plan
  Participants (2):
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1, device_id=GPU-b0c68926-7220-fe79-3599-5a4d2a232bfb))
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100))

  Programs (2):
    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:1, device_id=GPU-b0c68926-7220-fe79-3599-5a4d2a232bfb)) [3 instructions]:
      [0] Wait(id=0)
      [1] Pull(num_entries=1)
  [0] src_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000002, tensor_name=<none>, node_id=<none>), src_offset_bytes=0, dst_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000003, tensor_name=<none>, node_id=<none>), dst_offset_bytes=0, count=16, dtype=6, src_device=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100, src_ptr=0x0, dst_ptr=0x0
      [2] SyncPoint(id=1, wait_count=1)

    Participant(node_id=00000000-0000-0000-0000-000000000000, device=Device(torch_device=cuda:0, device_id=GPU-cf5a1501-a7d6-2c14-529d-2b91d5e23100)) [5 instructions]:
      [0] Copy(num_entries=1)
  [0] src_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000001, tensor_name=<none>, node_id=<none>), src_offset_bytes=0, dst_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000002, tensor_name=<none>, node_id=<none>), dst_offset_bytes=0, count=16, dtype=6, src_ptr=0x0, dst_ptr=0x0
      [1] SyncPoint(id=0, wait_count=2)
      [2] Wait(id=0)
      [3] Wait(id=1)
      [4] Copy(num_entries=1)
  [0] src_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000001, tensor_name=<none>, node_id=<none>), src_offset_bytes=0, dst_ref=ShardRef(shard_id=00000000-0000-0000-0000-000000000002, tensor_name=<none>, node_id=<none>), dst_offset_bytes=0, count=16, dtype=6, src_ptr=0x0, dst_ptr=0x0

)");
}

//==============================================================================
}  // namespace setu::test::native
//==============================================================================
