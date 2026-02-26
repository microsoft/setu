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
#include "commons/utils/PendingOperations.h"
//==============================================================================
namespace setu::test::native {
//==============================================================================
namespace {
//==============================================================================

using setu::commons::utils::PendingOperations;

// Void-payload type: WaiterId=string, BlockerId=int32_t
using VoidPending = PendingOperations<std::string, std::int32_t>;

// Payload type: WaiterId=string, BlockerId=string, Payload=int32_t
using PayloadPending =
    PendingOperations<std::string, std::string, std::int32_t>;

//==============================================================================
// N:1 Pattern Tests (many waiters on one blocker)
//==============================================================================

TEST(PendingOperationsTest, N1_RegisterAddResolve_ReturnsAllWaiters) {
  VoidPending pending;
  pending.RegisterBlocker(1);

  auto r1 = pending.AddWaiter("waiter_a", {1});
  auto r2 = pending.AddWaiter("waiter_b", {1});
  auto r3 = pending.AddWaiter("waiter_c", {1});

  EXPECT_FALSE(r1.has_value()) << "Waiter should be stored, not returned";
  EXPECT_FALSE(r2.has_value());
  EXPECT_FALSE(r3.has_value());

  auto unblocked = pending.Resolve(1);
  ASSERT_EQ(unblocked.size(), 3u);

  // Sort for deterministic comparison
  std::sort(unblocked.begin(), unblocked.end());
  EXPECT_EQ(unblocked[0], "waiter_a");
  EXPECT_EQ(unblocked[1], "waiter_b");
  EXPECT_EQ(unblocked[2], "waiter_c");
}

TEST(PendingOperationsTest, N1_ResolveWithNoWaiters_ReturnsEmpty) {
  VoidPending pending;
  pending.RegisterBlocker(1);

  auto unblocked = pending.Resolve(1);
  EXPECT_TRUE(unblocked.empty());
}

TEST(PendingOperationsTest, N1_ResolveUnregisteredKey_ReturnsEmpty) {
  VoidPending pending;

  auto unblocked = pending.Resolve(42);
  EXPECT_TRUE(unblocked.empty());
}

//==============================================================================
// 1:N Pattern Tests (one waiter blocked by many keys)
//==============================================================================

TEST(PendingOperationsTest, 1N_SingleWaiterMultipleBlockers_ReturnsOnLast) {
  PayloadPending pending;

  auto r = pending.AddWaiter("w1", {"op1", "op2", "op3"}, 100);
  EXPECT_FALSE(r.has_value());

  auto u1 = pending.Resolve("op1");
  EXPECT_TRUE(u1.empty()) << "Not all blockers resolved yet";

  auto u2 = pending.Resolve("op2");
  EXPECT_TRUE(u2.empty()) << "Not all blockers resolved yet";

  auto u3 = pending.Resolve("op3");
  ASSERT_EQ(u3.size(), 1u);
  EXPECT_EQ(u3[0], 100);
}

TEST(PendingOperationsTest, 1N_MultipleWaitersOverlappingBlockers) {
  PayloadPending pending;

  // Waiter "w1" blocked by {A, B}
  // Waiter "w2" blocked by {B, C}
  pending.AddWaiter("w1", {"A", "B"}, 100);
  pending.AddWaiter("w2", {"B", "C"}, 200);

  auto u_a = pending.Resolve("A");
  EXPECT_TRUE(u_a.empty()) << "w1 still blocked by B";

  auto u_c = pending.Resolve("C");
  EXPECT_TRUE(u_c.empty()) << "w2 still blocked by B";

  // Resolving B should unblock both waiters
  auto u_b = pending.Resolve("B");
  ASSERT_EQ(u_b.size(), 2u);

  std::sort(u_b.begin(), u_b.end());
  EXPECT_EQ(u_b[0], 100);
  EXPECT_EQ(u_b[1], 200);
}

//==============================================================================
// Late Arrival Tests
//==============================================================================

TEST(PendingOperationsTest, LateArrival_RegisterResolveAdd_ReturnsImmediately) {
  VoidPending pending;
  pending.RegisterBlocker(1);

  auto unblocked = pending.Resolve(1);
  EXPECT_TRUE(unblocked.empty());

  // Late arrival: AddWaiter after Resolve should return immediately
  auto result = pending.AddWaiter("late_waiter", {1});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "late_waiter");
}

TEST(PendingOperationsTest, LateArrival_MultipleRegistrations_DrainOneAtATime) {
  VoidPending pending;
  pending.RegisterBlocker(1);
  pending.RegisterBlocker(1);  // refcount = 2

  pending.Resolve(1);

  // First late arrival drains one registration
  auto r1 = pending.AddWaiter("late_1", {1});
  ASSERT_TRUE(r1.has_value());
  EXPECT_EQ(*r1, "late_1");

  // Second late arrival drains the other registration
  auto r2 = pending.AddWaiter("late_2", {1});
  ASSERT_TRUE(r2.has_value());
  EXPECT_EQ(*r2, "late_2");

  // Now registration count is 0, tombstone should be cleaned up.
  EXPECT_FALSE(pending.IsBlockerRegistered(1));
}

TEST(PendingOperationsTest, LateArrival_NotRegistered_NoTombstone) {
  VoidPending pending;

  // Resolve without Register — no tombstone created
  pending.Resolve(1);

  // AddWaiter should store (not return immediately) because there's no
  // tombstone for unregistered blockers
  auto result = pending.AddWaiter("waiter", {1});
  EXPECT_FALSE(result.has_value()) << "No tombstone for unregistered blocker";

  // Resolve again to release the stored waiter
  auto unblocked = pending.Resolve(1);
  ASSERT_EQ(unblocked.size(), 1u);
  EXPECT_EQ(unblocked[0], "waiter");
}

TEST(PendingOperationsTest, LateArrival_WithPayload_ReturnsPayload) {
  PayloadPending pending;
  pending.RegisterBlocker("key");

  pending.Resolve("key");

  auto result = pending.AddWaiter("w1", {"key"}, 42);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 42);
}

//==============================================================================
// IsBlockerRegistered Tests
//==============================================================================

TEST(PendingOperationsTest, IsBlockerRegistered_TrueAfterRegister) {
  VoidPending pending;
  pending.RegisterBlocker(1);
  EXPECT_TRUE(pending.IsBlockerRegistered(1));
}

TEST(PendingOperationsTest, IsBlockerRegistered_FalseForUnknownKey) {
  VoidPending pending;
  EXPECT_FALSE(pending.IsBlockerRegistered(42));
}

TEST(PendingOperationsTest, IsBlockerRegistered_TrueAfterResolve_UntilDrained) {
  VoidPending pending;
  pending.RegisterBlocker(1);

  pending.Resolve(1);

  // Registration persists until drained by late-arrival AddWaiter
  EXPECT_TRUE(pending.IsBlockerRegistered(1));

  // Drain via late arrival
  auto result = pending.AddWaiter("late", {1});
  ASSERT_TRUE(result.has_value());

  // Now registration is drained
  EXPECT_FALSE(pending.IsBlockerRegistered(1));
}

//==============================================================================
// RemoveBlocker Tests
//==============================================================================

TEST(PendingOperationsTest, RemoveBlocker_ClearsRegistrationAndTombstone) {
  VoidPending pending;
  pending.RegisterBlocker(1);
  pending.Resolve(1);

  // Tombstone exists — verify via late arrival
  pending.RemoveBlocker(1);

  EXPECT_FALSE(pending.IsBlockerRegistered(1));

  // After removal, tombstone is gone — AddWaiter stores the item
  auto result = pending.AddWaiter("new_waiter", {1});
  EXPECT_FALSE(result.has_value())
      << "Tombstone should be gone after RemoveBlocker";
}

TEST(PendingOperationsTest, RemoveBlocker_ClearsReverseIndex) {
  VoidPending pending;
  pending.RegisterBlocker(1);

  pending.AddWaiter("w1", {1});

  // RemoveBlocker clears the reverse index (does NOT release waiters)
  pending.RemoveBlocker(1);

  // Resolving after RemoveBlocker should return nothing (reverse index gone)
  auto unblocked = pending.Resolve(1);
  EXPECT_TRUE(unblocked.empty());
}

//==============================================================================
// RemoveWaiter Tests
//==============================================================================

TEST(PendingOperationsTest, RemoveWaiter_CleansUpWaiterAndReverseIndex) {
  VoidPending pending;

  pending.AddWaiter("w1", {1});
  pending.AddWaiter("w2", {1});

  // Remove w1 before resolving
  pending.RemoveWaiter("w1");

  // Resolve should only return w2
  auto unblocked = pending.Resolve(1);
  ASSERT_EQ(unblocked.size(), 1u);
  EXPECT_EQ(unblocked[0], "w2");
}

TEST(PendingOperationsTest, RemoveWaiter_NonexistentWaiter_NoOp) {
  VoidPending pending;

  // Should not crash
  pending.RemoveWaiter("does_not_exist");
}

TEST(PendingOperationsTest, RemoveWaiter_CleansUpMultipleBlockerRefs) {
  PayloadPending pending;

  pending.AddWaiter("w1", {"A", "B", "C"}, 100);

  pending.RemoveWaiter("w1");

  // Resolving any of the blockers should return nothing
  EXPECT_TRUE(pending.Resolve("A").empty());
  EXPECT_TRUE(pending.Resolve("B").empty());
  EXPECT_TRUE(pending.Resolve("C").empty());
}

//==============================================================================
// Mixed Pattern Tests
//==============================================================================

TEST(PendingOperationsTest, Mixed_N1And1N_Coexist) {
  PayloadPending pending;

  // N:1 pattern: multiple waiters blocked by "alpha"
  pending.AddWaiter("w1", {"alpha"}, 1);
  pending.AddWaiter("w2", {"alpha"}, 2);

  // 1:N pattern: one waiter blocked by multiple keys
  pending.AddWaiter("w3", {"alpha", "beta"}, 3);

  // Resolve alpha: w1, w2 unblocked; w3 still blocked by beta
  auto u_alpha = pending.Resolve("alpha");
  ASSERT_EQ(u_alpha.size(), 2u);
  std::sort(u_alpha.begin(), u_alpha.end());
  EXPECT_EQ(u_alpha[0], 1);
  EXPECT_EQ(u_alpha[1], 2);

  // Resolve beta: w3 now fully unblocked
  auto u_beta = pending.Resolve("beta");
  ASSERT_EQ(u_beta.size(), 1u);
  EXPECT_EQ(u_beta[0], 3);
}

TEST(PendingOperationsTest, Mixed_ResolveIdempotent) {
  PayloadPending pending;

  pending.AddWaiter("w1", {"key"}, 42);
  auto u1 = pending.Resolve("key");
  ASSERT_EQ(u1.size(), 1u);
  EXPECT_EQ(u1[0], 42);

  // Second resolve of same key returns nothing
  auto u2 = pending.Resolve("key");
  EXPECT_TRUE(u2.empty());
}

TEST(PendingOperationsTest, Mixed_VoidPayload_ReturnsWaiterIds) {
  VoidPending pending;

  pending.AddWaiter("client_1", {10});
  pending.AddWaiter("client_2", {10});

  auto unblocked = pending.Resolve(10);
  ASSERT_EQ(unblocked.size(), 2u);

  std::sort(unblocked.begin(), unblocked.end());
  EXPECT_EQ(unblocked[0], "client_1");
  EXPECT_EQ(unblocked[1], "client_2");
}

TEST(PendingOperationsTest, Mixed_PartialBlockersResolved_ViaRegistration) {
  PayloadPending pending;
  pending.RegisterBlocker("x");

  // Resolve "x" (creates tombstone because registered)
  pending.Resolve("x");

  // Waiter has blockers {x, y} — x is resolved (tombstoned) but y is not
  auto result = pending.AddWaiter("w1", {"x", "y"}, 50);
  EXPECT_FALSE(result.has_value()) << "y is still unresolved";

  // Now resolve y
  auto unblocked = pending.Resolve("y");
  ASSERT_EQ(unblocked.size(), 1u);
  EXPECT_EQ(unblocked[0], 50);
}

TEST(PendingOperationsTest, Mixed_AllBlockersResolved_ViaRegistration) {
  PayloadPending pending;
  pending.RegisterBlocker("x");
  pending.RegisterBlocker("y");

  pending.Resolve("x");
  pending.Resolve("y");

  // All blockers resolved — returns immediately
  auto result = pending.AddWaiter("w1", {"x", "y"}, 99);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 99);
}

//==============================================================================
}  // namespace
//==============================================================================
}  // namespace setu::test::native
//==============================================================================
