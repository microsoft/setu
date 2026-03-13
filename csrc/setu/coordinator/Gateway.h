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
#include "commons/utils/ZmqHelper.h"
#include "coordinator/Types.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
//==============================================================================

/// @brief Owns the ZMQ socket and handles all network I/O for the Coordinator.
///
/// Gateway runs on a dedicated thread and bridges NodeAgent messages to/from
/// the internal inbox/outbox queues. This keeps ZMQ isolated to one thread.
class Gateway {
 public:
  Gateway(std::shared_ptr<zmq::context_t> zmq_context, std::size_t port,
          Queue<InboxMessage>& inbox_queue, Queue<OutboxMessage>& outbox_queue);
  ~Gateway();

  void Start();
  void Stop();

  /// @brief Wake the Gateway from its poll() call so it drains the outbox
  /// immediately. Thread-safe: can be called from any thread.
  void NotifyOutbox();

 private:
  void InitSockets();
  void CloseSockets();
  void Loop();

  std::shared_ptr<zmq::context_t> zmq_context_;
  std::size_t port_;

  Queue<InboxMessage>& inbox_queue_;
  Queue<OutboxMessage>& outbox_queue_;

  ZmqSocketPtr node_agent_socket_;

  /// Inproc PAIR sockets for self-pipe wakeup pattern.
  /// When a producer pushes to outbox_queue_, it sends a byte on
  /// wakeup_send_ which causes zmq::poll() (watching wakeup_recv_) to
  /// return immediately.
  ZmqSocketPtr wakeup_recv_;
  ZmqSocketPtr wakeup_send_;
  std::mutex wakeup_mutex_;  // Serializes NotifyOutbox() from Handler/Executor

  std::thread thread_;
  std::atomic<bool> running_{false};
};
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
