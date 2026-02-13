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
#include "commons/BoostCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/Device.h"
#include "commons/datatypes/TensorDimSpec.h"
#include "commons/datatypes/TensorShard.h"
#include "commons/datatypes/TensorShardHandle.h"
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/datatypes/TensorShardSpec.h"
//==============================================================================
using setu::commons::GenerateUUID;
using setu::commons::NodeId;
using setu::commons::datatypes::Device;
using setu::commons::datatypes::TensorDimSpec;
using setu::commons::datatypes::TensorShard;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorShardPtr;
using setu::commons::datatypes::TensorShardReadHandle;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::datatypes::TensorShardWriteHandle;
//==============================================================================

class TensorShardHandleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a unique temp directory for lock files per test
    lock_base_dir_ = std::filesystem::temp_directory_path() /
                     ("setu_test_locks_" + std::to_string(getpid()) + "_" +
                      std::to_string(test_counter_++));
    std::filesystem::create_directories(lock_base_dir_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(lock_base_dir_, ec);
  }

  [[nodiscard]] TensorShardPtr MakeShard(
      const std::string& name = "test_tensor", std::int64_t size = 16) const {
    std::vector<TensorDimSpec> dims;
    dims.emplace_back("x", static_cast<std::size_t>(size), 0, size);
    TensorShardSpec spec(name, dims, torch::kFloat32, Device(torch::kCPU));
    TensorShardMetadata metadata(spec, GenerateUUID());
    torch::Tensor tensor = torch::ones({size}, torch::kFloat32);
    return std::make_shared<TensorShard>(std::move(metadata), std::move(tensor),
                                         lock_base_dir_.string());
  }

  std::filesystem::path lock_base_dir_;

 private:
  static std::uint64_t test_counter_;
};

std::uint64_t TensorShardHandleTest::test_counter_ = 0;

//==============================================================================
// TensorShard Construction
//==============================================================================

TEST_F(TensorShardHandleTest, Constructor_ValidParams_CreatesLockFile) {
  auto shard = MakeShard();

  // Lock file should exist at {base_dir}/{uuid}.lock
  const auto expected_lock_path =
      lock_base_dir_ / (boost::uuids::to_string(shard->metadata.id) + ".lock");
  EXPECT_TRUE(std::filesystem::exists(expected_lock_path))
      << "Lock file should be created at: " << expected_lock_path;
}

TEST_F(TensorShardHandleTest, Constructor_InvalidTensor_Throws) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("x", 16, 0, 16);
  TensorShardSpec spec("test", dims, torch::kFloat32, Device(torch::kCPU));
  TensorShardMetadata metadata(spec, GenerateUUID());

  torch::Tensor empty_tensor;
  EXPECT_THROW(
      {
        TensorShard(std::move(metadata), std::move(empty_tensor),
                    lock_base_dir_.string());
      },
      std::runtime_error)
      << "Should throw on undefined tensor";
}

TEST_F(TensorShardHandleTest, Constructor_EmptyLockBaseDir_Throws) {
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("x", 16, 0, 16);
  TensorShardSpec spec("test", dims, torch::kFloat32, Device(torch::kCPU));
  TensorShardMetadata metadata(spec, GenerateUUID());
  torch::Tensor tensor = torch::ones({16}, torch::kFloat32);

  EXPECT_THROW(
      { TensorShard(std::move(metadata), std::move(tensor), ""); },
      std::invalid_argument)
      << "Should throw on empty lock_base_dir";
}

TEST_F(TensorShardHandleTest, Constructor_SameShardId_SameLockFile) {
  // Two shards with the same metadata ID should produce the same lock file
  std::vector<TensorDimSpec> dims;
  dims.emplace_back("x", 16, 0, 16);
  TensorShardSpec spec("test", dims, torch::kFloat32, Device(torch::kCPU));

  auto shared_id = GenerateUUID();
  NodeId owner = GenerateUUID();
  TensorShardMetadata meta1(shared_id, spec, owner);
  TensorShardMetadata meta2(shared_id, spec, owner);

  torch::Tensor tensor1 = torch::ones({16}, torch::kFloat32);
  torch::Tensor tensor2 = torch::ones({16}, torch::kFloat32);

  auto shard1 = std::make_shared<TensorShard>(
      std::move(meta1), std::move(tensor1), lock_base_dir_.string());
  auto shard2 = std::make_shared<TensorShard>(
      std::move(meta2), std::move(tensor2), lock_base_dir_.string());

  // Both should reference the same lock file
  const auto expected_lock_path =
      lock_base_dir_ / (boost::uuids::to_string(shared_id) + ".lock");

  // Only one lock file should exist for that UUID
  std::size_t lock_file_count = 0;
  for (const auto& entry :
       std::filesystem::directory_iterator(lock_base_dir_)) {
    if (entry.path().extension() == ".lock") {
      ++lock_file_count;
    }
  }
  EXPECT_EQ(lock_file_count, 1)
      << "Two shards with same ID should share one lock file";
}

//==============================================================================
// Read Handle
//==============================================================================

TEST_F(TensorShardHandleTest, ReadHandle_AcquiresAndReleases) {
  auto shard = MakeShard();

  {
    TensorShardReadHandle read_handle(shard);
    EXPECT_NE(read_handle.GetDevicePtr(), nullptr)
        << "Device pointer should be valid under read lock";
    EXPECT_EQ(read_handle.GetShard(), shard.get())
        << "GetShard should return the underlying shard";
  }
  // Lock released here — no crash means success
}

TEST_F(TensorShardHandleTest, ReadHandle_NullShard_Throws) {
  TensorShardPtr null_shard = nullptr;
  EXPECT_THROW(
      { [[maybe_unused]] TensorShardReadHandle rh(null_shard); },
      std::invalid_argument)
      << "Should throw on null shard";
}

TEST_F(TensorShardHandleTest, ReadHandle_MultipleReaders_Concurrent) {
  auto shard = MakeShard();

  // Multiple read handles should coexist without deadlock
  TensorShardReadHandle read1(shard);
  TensorShardReadHandle read2(shard);
  TensorShardReadHandle read3(shard);

  EXPECT_EQ(read1.GetDevicePtr(), read2.GetDevicePtr());
  EXPECT_EQ(read2.GetDevicePtr(), read3.GetDevicePtr());
}

TEST_F(TensorShardHandleTest, ReadHandle_DevicePtrMatchesTensor) {
  auto shard = MakeShard();

  TensorShardReadHandle read_handle(shard);
  EXPECT_EQ(read_handle.GetDevicePtr(), shard->tensor.data_ptr())
      << "Device pointer should match the tensor's data pointer";
}

//==============================================================================
// Write Handle
//==============================================================================

TEST_F(TensorShardHandleTest, WriteHandle_AcquiresAndReleases) {
  auto shard = MakeShard();

  {
    TensorShardWriteHandle write_handle(shard);
    EXPECT_NE(write_handle.GetDevicePtr(), nullptr)
        << "Device pointer should be valid under write lock";
    EXPECT_EQ(write_handle.GetShard(), shard.get())
        << "GetShard should return the underlying shard";
  }
  // Lock released here — no crash means success
}

TEST_F(TensorShardHandleTest, WriteHandle_NullShard_Throws) {
  TensorShardPtr null_shard = nullptr;
  EXPECT_THROW(
      { [[maybe_unused]] TensorShardWriteHandle wh(null_shard); },
      std::invalid_argument)
      << "Should throw on null shard";
}

TEST_F(TensorShardHandleTest, WriteHandle_ThenReadHandle_Sequential) {
  auto shard = MakeShard();

  // Write then read sequentially — should not deadlock
  {
    TensorShardWriteHandle write_handle(shard);
    auto* ptr = static_cast<float*>(write_handle.GetDevicePtr());
    ptr[0] = 42.0f;
  }

  {
    TensorShardReadHandle read_handle(shard);
    const auto* ptr = static_cast<const float*>(read_handle.GetDevicePtr());
    EXPECT_FLOAT_EQ(ptr[0], 42.0f)
        << "Value written under write lock should be readable";
  }
}

TEST_F(TensorShardHandleTest, ReadHandle_ThenWriteHandle_Sequential) {
  auto shard = MakeShard();

  {
    TensorShardReadHandle read_handle(shard);
    EXPECT_NE(read_handle.GetDevicePtr(), nullptr);
  }

  {
    TensorShardWriteHandle write_handle(shard);
    EXPECT_NE(write_handle.GetDevicePtr(), nullptr);
  }
}

//==============================================================================
// Cross-Thread Tests
//==============================================================================

TEST_F(TensorShardHandleTest, ReadHandle_MultipleThreads_NoCrash) {
  auto shard = MakeShard();
  constexpr std::size_t kNumThreads = 8;

  std::vector<std::thread> threads;
  std::atomic<std::size_t> completed{0};

  for (std::size_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&shard, &completed]() {
      TensorShardReadHandle read_handle(shard);
      EXPECT_NE(read_handle.GetDevicePtr(), nullptr);
      // Hold lock briefly
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      completed.fetch_add(1, std::memory_order_relaxed);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(completed.load(), kNumThreads)
      << "All reader threads should complete";
}

TEST_F(TensorShardHandleTest, WriteHandle_ExclusiveAccess_CrossThread) {
  auto shard = MakeShard("counter_tensor", 1);
  constexpr std::size_t kNumThreads = 4;
  constexpr std::size_t kIncrementsPerThread = 100;

  // Each thread acquires a write lock and increments a counter
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&shard]() {
      for (std::size_t j = 0; j < kIncrementsPerThread; ++j) {
        TensorShardWriteHandle write_handle(shard);
        auto* ptr = static_cast<float*>(write_handle.GetDevicePtr());
        float val = ptr[0];
        // Yield to increase chance of race if locking is broken
        std::this_thread::yield();
        ptr[0] = val + 1.0f;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  TensorShardReadHandle read_handle(shard);
  const auto* ptr = static_cast<const float*>(read_handle.GetDevicePtr());
  EXPECT_FLOAT_EQ(ptr[0], 1.0f + kNumThreads * kIncrementsPerThread)
      << "Counter should reflect all increments (no lost updates)";
}

//==============================================================================
// Cross-Process Tests
//==============================================================================

TEST_F(TensorShardHandleTest, WriteLock_BlocksReadLock_CrossProcess) {
  auto shard = MakeShard();
  const auto lock_path =
      lock_base_dir_ / (boost::uuids::to_string(shard->metadata.id) + ".lock");

  // Pipes for parent-child synchronization
  std::int32_t pipe_parent_to_child[2];
  std::int32_t pipe_child_to_parent[2];
  ASSERT_EQ(pipe(pipe_parent_to_child), 0);
  ASSERT_EQ(pipe(pipe_child_to_parent), 0);

  pid_t pid = fork();
  ASSERT_NE(pid, -1) << "fork() failed";

  if (pid == 0) {
    // ── Child process ──
    close(pipe_parent_to_child[1]);
    close(pipe_child_to_parent[0]);

    // Wait for parent to signal that write lock is held
    char buf;
    ASSERT_EQ(read(pipe_parent_to_child[0], &buf, 1), static_cast<ssize_t>(1));

    // Try to acquire a shared (read) lock — should block
    auto file_lock =
        std::make_shared<setu::commons::FileLock>(lock_path.string().c_str());
    bool acquired = file_lock->try_lock_sharable();

    // Signal parent whether we could acquire the lock
    char result = acquired ? '1' : '0';
    ASSERT_EQ(write(pipe_child_to_parent[1], &result, 1),
              static_cast<ssize_t>(1));

    if (acquired) {
      file_lock->unlock_sharable();
    }

    close(pipe_parent_to_child[0]);
    close(pipe_child_to_parent[1]);
    _exit(0);
  }

  // ── Parent process ──
  close(pipe_parent_to_child[0]);
  close(pipe_child_to_parent[1]);

  {
    // Acquire exclusive write lock
    TensorShardWriteHandle write_handle(shard);

    // Signal child to try read lock
    char signal = 'g';
    ASSERT_EQ(write(pipe_parent_to_child[1], &signal, 1),
              static_cast<ssize_t>(1));

    // Give child time to attempt lock
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Read child's result
    char result;
    ASSERT_EQ(read(pipe_child_to_parent[0], &result, 1),
              static_cast<ssize_t>(1));
    EXPECT_EQ(result, '0')
        << "Child should NOT be able to acquire read lock while parent holds "
           "write lock";
  }

  close(pipe_parent_to_child[1]);
  close(pipe_child_to_parent[0]);

  std::int32_t status;
  waitpid(pid, &status, 0);
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

TEST_F(TensorShardHandleTest, ReadLock_AllowsMultipleReaders_CrossProcess) {
  auto shard = MakeShard();
  const auto lock_path =
      lock_base_dir_ / (boost::uuids::to_string(shard->metadata.id) + ".lock");

  std::int32_t pipe_parent_to_child[2];
  std::int32_t pipe_child_to_parent[2];
  ASSERT_EQ(pipe(pipe_parent_to_child), 0);
  ASSERT_EQ(pipe(pipe_child_to_parent), 0);

  pid_t pid = fork();
  ASSERT_NE(pid, -1) << "fork() failed";

  if (pid == 0) {
    // ── Child process ──
    close(pipe_parent_to_child[1]);
    close(pipe_child_to_parent[0]);

    // Wait for parent to signal that read lock is held
    char buf;
    ASSERT_EQ(read(pipe_parent_to_child[0], &buf, 1), static_cast<ssize_t>(1));

    // Try to acquire a shared (read) lock — should succeed
    auto file_lock =
        std::make_shared<setu::commons::FileLock>(lock_path.string().c_str());
    bool acquired = file_lock->try_lock_sharable();

    char result = acquired ? '1' : '0';
    ASSERT_EQ(write(pipe_child_to_parent[1], &result, 1),
              static_cast<ssize_t>(1));

    if (acquired) {
      file_lock->unlock_sharable();
    }

    close(pipe_parent_to_child[0]);
    close(pipe_child_to_parent[1]);
    _exit(0);
  }

  // ── Parent process ──
  close(pipe_parent_to_child[0]);
  close(pipe_child_to_parent[1]);

  {
    // Acquire shared read lock
    TensorShardReadHandle read_handle(shard);

    // Signal child to try read lock
    char signal = 'g';
    ASSERT_EQ(write(pipe_parent_to_child[1], &signal, 1),
              static_cast<ssize_t>(1));

    // Read child's result
    char result;
    ASSERT_EQ(read(pipe_child_to_parent[0], &result, 1),
              static_cast<ssize_t>(1));
    EXPECT_EQ(result, '1')
        << "Child SHOULD be able to acquire read lock while parent holds "
           "read lock";
  }

  close(pipe_parent_to_child[1]);
  close(pipe_child_to_parent[0]);

  std::int32_t status;
  waitpid(pid, &status, 0);
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

TEST_F(TensorShardHandleTest, ReadLock_BlocksWriteLock_CrossProcess) {
  auto shard = MakeShard();
  const auto lock_path =
      lock_base_dir_ / (boost::uuids::to_string(shard->metadata.id) + ".lock");

  std::int32_t pipe_parent_to_child[2];
  std::int32_t pipe_child_to_parent[2];
  ASSERT_EQ(pipe(pipe_parent_to_child), 0);
  ASSERT_EQ(pipe(pipe_child_to_parent), 0);

  pid_t pid = fork();
  ASSERT_NE(pid, -1) << "fork() failed";

  if (pid == 0) {
    // ── Child process ──
    close(pipe_parent_to_child[1]);
    close(pipe_child_to_parent[0]);

    char buf;
    ASSERT_EQ(read(pipe_parent_to_child[0], &buf, 1), static_cast<ssize_t>(1));

    // Try to acquire exclusive (write) lock — should block
    auto file_lock =
        std::make_shared<setu::commons::FileLock>(lock_path.string().c_str());
    bool acquired = file_lock->try_lock();

    char result = acquired ? '1' : '0';
    ASSERT_EQ(write(pipe_child_to_parent[1], &result, 1),
              static_cast<ssize_t>(1));

    if (acquired) {
      file_lock->unlock();
    }

    close(pipe_parent_to_child[0]);
    close(pipe_child_to_parent[1]);
    _exit(0);
  }

  // ── Parent process ──
  close(pipe_parent_to_child[0]);
  close(pipe_child_to_parent[1]);

  {
    TensorShardReadHandle read_handle(shard);

    char signal = 'g';
    ASSERT_EQ(write(pipe_parent_to_child[1], &signal, 1),
              static_cast<ssize_t>(1));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    char result;
    ASSERT_EQ(read(pipe_child_to_parent[0], &result, 1),
              static_cast<ssize_t>(1));
    EXPECT_EQ(result, '0')
        << "Child should NOT be able to acquire write lock while parent holds "
           "read lock";
  }

  close(pipe_parent_to_child[1]);
  close(pipe_child_to_parent[0]);

  std::int32_t status;
  waitpid(pid, &status, 0);
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

//==============================================================================
