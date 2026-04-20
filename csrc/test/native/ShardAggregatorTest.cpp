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
#include "commons/utils/ShardAggregator.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
namespace {
//==============================================================================

using setu::commons::GenerateUUID;
using setu::commons::Identity;
using setu::commons::RequestId;
using setu::commons::ShardId;
using setu::commons::utils::AggregationParticipant;
using setu::commons::utils::CancelledGroup;
using setu::commons::utils::CompletedGroup;
using setu::commons::utils::ShardAggregator;

using Aggregator = ShardAggregator<std::string, std::int32_t>;

auto kAlwaysValid = [](std::int32_t, std::int32_t) { return true; };
auto kAlwaysReject = [](std::int32_t, std::int32_t) { return false; };

AggregationParticipant MakeParticipant(const std::string& id) {
  return AggregationParticipant{Identity{id}, RequestId{}};
}

template <typename T, typename Outcome>
bool Holds(const Outcome& outcome) {
  return std::holds_alternative<T>(outcome);
}

using Completed = CompletedGroup<std::int32_t>;

//==============================================================================
// Baseline: one group, all shards arrive → CompletedGroup.
//==============================================================================

TEST(ShardAggregatorTest, SingleGroupCompletesBaseline) {
  Aggregator agg;
  constexpr std::size_t kExpected = 3;
  const auto s1 = GenerateUUID();
  const auto s2 = GenerateUUID();
  const auto s3 = GenerateUUID();

  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s1, 42,
                                                MakeParticipant("p1"),
                                                kExpected, kAlwaysValid)));
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s2, 42,
                                                MakeParticipant("p2"),
                                                kExpected, kAlwaysValid)));

  auto r3 =
      agg.Submit("K", s3, 42, MakeParticipant("p3"), kExpected, kAlwaysValid);
  ASSERT_TRUE(Holds<Completed>(r3));
  auto& completed = std::get<Completed>(r3);
  EXPECT_EQ(completed.payload, 42);
  EXPECT_EQ(completed.participants.size(), 3u);
}

//==============================================================================
// A duplicate shard_id for the same key opens a new FIFO group.
//==============================================================================

TEST(ShardAggregatorTest, DuplicateShardOpensNewGroup) {
  Aggregator agg;
  constexpr std::size_t kExpected = 2;
  const auto s1 = GenerateUUID();
  const auto s2 = GenerateUUID();

  // Group 1 opens with shard 1 (peer shard 2 not yet).
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s1, 10,
                                                MakeParticipant("p1_op1"),
                                                kExpected, kAlwaysValid)));

  // Shard 1 arrives again for op 2 — must land in a NEW tail group.
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s1, 20,
                                                MakeParticipant("p1_op2"),
                                                kExpected, kAlwaysValid)));

  // Shard 2 arrives for op 1 → completes group 1.
  auto r3 = agg.Submit("K", s2, 10, MakeParticipant("p2_op1"), kExpected,
                      kAlwaysValid);
  ASSERT_TRUE(Holds<Completed>(r3));
  EXPECT_EQ(std::get<Completed>(r3).payload, 10);
  EXPECT_EQ(std::get<Completed>(r3).participants.size(), 2u);

  // Shard 2 arrives for op 2 → completes group 2.
  auto r4 = agg.Submit("K", s2, 20, MakeParticipant("p2_op2"), kExpected,
                      kAlwaysValid);
  ASSERT_TRUE(Holds<Completed>(r4));
  EXPECT_EQ(std::get<Completed>(r4).payload, 20);
  EXPECT_EQ(std::get<Completed>(r4).participants.size(), 2u);
}

//==============================================================================
// A group in the middle of the list completes first; neighbors survive.
//==============================================================================

TEST(ShardAggregatorTest, MiddleGroupCompletesFirst) {
  Aggregator agg;
  constexpr std::size_t kExpected = 2;
  const auto s1 = GenerateUUID();
  const auto s2 = GenerateUUID();

  // Open 3 groups by submitting shard 1 three times.
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s1, 100,
                                                MakeParticipant("p1_op1"),
                                                kExpected, kAlwaysValid)));
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s1, 200,
                                                MakeParticipant("p1_op2"),
                                                kExpected, kAlwaysValid)));
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s1, 300,
                                                MakeParticipant("p1_op3"),
                                                kExpected, kAlwaysValid)));

  // Peer shard 2 arrives three times — each lands in the head-most open
  // group missing shard 2.
  auto r_op1 = agg.Submit("K", s2, 100, MakeParticipant("p2_op1"), kExpected,
                         kAlwaysValid);
  ASSERT_TRUE(Holds<Completed>(r_op1));
  EXPECT_EQ(std::get<Completed>(r_op1).payload, 100);

  // Op 2 is now the head. Completing it exercises erase-of-non-tail on a
  // list with a tail element still present.
  auto r_op2 = agg.Submit("K", s2, 200, MakeParticipant("p2_op2"), kExpected,
                         kAlwaysValid);
  ASSERT_TRUE(Holds<Completed>(r_op2));
  EXPECT_EQ(std::get<Completed>(r_op2).payload, 200);

  auto r_op3 = agg.Submit("K", s2, 300, MakeParticipant("p2_op3"), kExpected,
                         kAlwaysValid);
  ASSERT_TRUE(Holds<Completed>(r_op3));
  EXPECT_EQ(std::get<Completed>(r_op3).payload, 300);
}

//==============================================================================
// Validation failure pops the specific group only; other open groups untouched.
//==============================================================================

TEST(ShardAggregatorTest, ValidationFailurePopsSpecificGroup) {
  Aggregator agg;
  constexpr std::size_t kExpected = 2;
  const auto s1 = GenerateUUID();
  const auto s2 = GenerateUUID();

  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s1, 10,
                                                MakeParticipant("p1_op1"),
                                                kExpected, kAlwaysValid)));
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s1, 20,
                                                MakeParticipant("p1_op2"),
                                                kExpected, kAlwaysValid)));

  // Shard 2 arrives for op 1 but validation rejects.
  auto rejected = agg.Submit("K", s2, 999, MakeParticipant("p2_op1"),
                            kExpected, kAlwaysReject);
  ASSERT_TRUE(Holds<CancelledGroup>(rejected));
  auto& c = std::get<CancelledGroup>(rejected);
  ASSERT_EQ(c.participants.size(), 1u);
  EXPECT_EQ(c.participants[0].identity, "p1_op1");

  // Group 2 is still alive and completable.
  auto completed = agg.Submit("K", s2, 20, MakeParticipant("p2_op2"),
                             kExpected, kAlwaysValid);
  ASSERT_TRUE(Holds<Completed>(completed));
  EXPECT_EQ(std::get<Completed>(completed).payload, 20);
}

//==============================================================================
// Cancel(key) flattens participants from every open group for that key only.
//==============================================================================

TEST(ShardAggregatorTest, CancelKeyFlattensAllGroupsForThatKey) {
  Aggregator agg;
  constexpr std::size_t kExpected = 2;
  const auto s1 = GenerateUUID();
  const auto s2 = GenerateUUID();

  // 3 open groups under key "A".
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("A", s1, 1,
                                                MakeParticipant("A_g1"),
                                                kExpected, kAlwaysValid)));
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("A", s1, 2,
                                                MakeParticipant("A_g2"),
                                                kExpected, kAlwaysValid)));
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("A", s1, 3,
                                                MakeParticipant("A_g3"),
                                                kExpected, kAlwaysValid)));

  // 1 open group under key "B".
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("B", s1, 99,
                                                MakeParticipant("B_g1"),
                                                kExpected, kAlwaysValid)));

  auto cancelled = agg.Cancel("A");
  ASSERT_EQ(cancelled.size(), 3u);

  std::set<Identity> identities;
  for (const auto& p : cancelled) identities.insert(p.identity);
  EXPECT_EQ(identities, (std::set<Identity>{"A_g1", "A_g2", "A_g3"}));

  // B still alive — completing it should succeed.
  auto b_done = agg.Submit("B", s2, 99, MakeParticipant("B_g1_peer"),
                          kExpected, kAlwaysValid);
  EXPECT_TRUE(Holds<Completed>(b_done));
}

//==============================================================================
// CancelIf wipes every open group for keys matching the predicate.
//==============================================================================

TEST(ShardAggregatorTest, CancelIfWipesAllMatchingKeys) {
  Aggregator agg;
  constexpr std::size_t kExpected = 2;
  const auto s1 = GenerateUUID();
  const auto s2 = GenerateUUID();

  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("A", s1, 1,
                                                MakeParticipant("A1"),
                                                kExpected, kAlwaysValid)));
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("A", s1, 2,
                                                MakeParticipant("A2"),
                                                kExpected, kAlwaysValid)));
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("B", s1, 1,
                                                MakeParticipant("B1"),
                                                kExpected, kAlwaysValid)));
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("C", s1, 1,
                                                MakeParticipant("C1"),
                                                kExpected, kAlwaysValid)));

  auto cancelled = agg.CancelIf(
      [](const std::string& k) { return k == "A" || k == "C"; });

  std::set<Identity> identities;
  for (const auto& p : cancelled) identities.insert(p.identity);
  EXPECT_EQ(identities, (std::set<Identity>{"A1", "A2", "C1"}));

  // B untouched — still completable.
  auto b_done = agg.Submit("B", s2, 1, MakeParticipant("B1_peer"), kExpected,
                          kAlwaysValid);
  EXPECT_TRUE(Holds<Completed>(b_done));
}

//==============================================================================
// expected_count must be stable for the open group targeted by a submission.
//==============================================================================

TEST(ShardAggregatorTest, ExpectedCountMismatchThrows) {
  Aggregator agg;
  const auto s1 = GenerateUUID();
  const auto s2 = GenerateUUID();

  EXPECT_TRUE(Holds<std::monostate>(
      agg.Submit("K", s1, 1, MakeParticipant("p1"), 3, kAlwaysValid)));

  EXPECT_THROW(
      {
        (void)agg.Submit("K", s2, 1, MakeParticipant("p2"), 4, kAlwaysValid);
      },
      std::exception);
}

//==============================================================================
// Erase at both head and tail of the list works.
//==============================================================================

TEST(ShardAggregatorTest, CompletionAtListHeadAndTail) {
  Aggregator agg;
  constexpr std::size_t kExpected = 2;
  const auto s1 = GenerateUUID();
  const auto s2 = GenerateUUID();

  // Open two groups.
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s1, 100,
                                                MakeParticipant("head_p1"),
                                                kExpected, kAlwaysValid)));
  EXPECT_TRUE(Holds<std::monostate>(agg.Submit("K", s1, 200,
                                                MakeParticipant("tail_p1"),
                                                kExpected, kAlwaysValid)));

  // First peer shard lands in the head group → completes head (erase head).
  auto head_done = agg.Submit("K", s2, 100, MakeParticipant("head_p2"),
                             kExpected, kAlwaysValid);
  ASSERT_TRUE(Holds<Completed>(head_done));
  EXPECT_EQ(std::get<Completed>(head_done).payload, 100);

  // Second peer shard lands in what's now the only group → completes tail
  // (erase of single remaining element, exercises erase-at-tail path).
  auto tail_done = agg.Submit("K", s2, 200, MakeParticipant("tail_p2"),
                             kExpected, kAlwaysValid);
  ASSERT_TRUE(Holds<Completed>(tail_done));
  EXPECT_EQ(std::get<Completed>(tail_done).payload, 200);
}

//==============================================================================
}  // namespace
}  // namespace setu::test::native
//==============================================================================
