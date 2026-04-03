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
#include "commons/Types.h"
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorSelection.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/enums/Enums.h"
#include "commons/utils/TorchTensorIPC.h"
#include "commons/utils/ZmqHelper.h"
#include "commons/utils/ring/CompletionEntry.h"
#include "commons/utils/ring/ShmRing.h"
#include "messaging/GetTensorHandleResponse.h"
#include "planner/hints/Hint.h"

namespace setu::client {
using setu::commons::CopyOperationId;
using setu::commons::ShardId;
using setu::commons::TensorName;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::TensorSelectionPtr;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardRefPtr;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::GetTensorHandleResponse;
using setu::commons::utils::TensorIPCSpec;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
using setu::commons::utils::ring::CompletionEntry;
using setu::commons::utils::ring::CompletionRingConsumer;
using setu::planner::hints::CompilerHint;

class Client {
 public:
  Client();
  ~Client();

  void Connect(const std::string& endpoint);

  void Disconnect();

  bool IsConnected() const;

  const std::string& GetEndpoint() const;

  std::optional<TensorShardRef> RegisterTensorShard(
      const TensorShardSpec& shard_spec);

  std::uint64_t SubmitCopy(const CopySpec& copy_spec,
                          const std::vector<CompilerHint>& hints = {});

  std::uint64_t SubmitPull(const CopySpec& copy_spec,
                           const std::vector<CompilerHint>& hints = {});

  CopyOperationId WaitForCopy(std::uint64_t local_id);

  void WaitForShardAllocation(ShardId shard_id);

  GetTensorHandleResponse GetTensorHandle(const TensorShardRef& shard_ref);

  TensorSelectionPtr Select(const TensorName& name);

  [[nodiscard]] std::vector<TensorShardRefPtr> GetShards() const;

  /// @brief Non-blocking poll for completed copy operations.
  /// Drains entries from the shared-memory completion ring.
  /// Returns (local_id, global CopyOperationId) pairs.
  struct Completion {
    std::uint64_t local_id;
    CopyOperationId copy_op_id;
  };
  [[nodiscard]] std::vector<Completion> PollCompletions();

 private:
  static constexpr std::int32_t kDeregisterTimeoutMs = 300000;
  static constexpr std::uint32_t kMaxPollBatch = 64;

  [[nodiscard]] bool DeregisterShards();

  // Zmq context and sockets
  ZmqContextPtr zmq_context_;
  std::string client_id_;
  ZmqSocketPtr query_socket_;
  ZmqSocketPtr submit_socket_;

  // Monotonic local ID counter for async submit operations
  std::atomic<std::uint64_t> next_local_id_{1};

  // Map from tensor name to list of shard refs owned by this client
  std::map<TensorName, std::vector<TensorShardRefPtr>> tensor_shards_;

  std::string endpoint_;
  bool is_connected_{false};

  // Completion ring (shared memory)
  void* completion_ring_mmap_{nullptr};
  std::size_t completion_ring_size_{0};
  std::string completion_ring_shm_name_;
  std::unique_ptr<CompletionRingConsumer> completion_ring_;
  std::unordered_map<std::uint64_t, CopyOperationId> completed_ops_;
};
//==============================================================================
}  // namespace setu::client
