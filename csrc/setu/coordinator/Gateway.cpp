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
#include "coordinator/Gateway.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/utils/Comm.h"
#include "commons/utils/ThreadingUtils.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::utils::Comm;
using setu::commons::utils::ZmqHelper;
//==============================================================================
constexpr std::int32_t kPollTimeoutMs = 100;
//==============================================================================
Gateway::Gateway(std::shared_ptr<zmq::context_t> zmq_context, std::size_t port,
                 Queue<InboxMessage>& inbox_queue,
                 Queue<OutboxMessage>& outbox_queue)
    : zmq_context_(zmq_context),
      port_(port),
      inbox_queue_(inbox_queue),
      outbox_queue_(outbox_queue) {
  InitSockets();
}

Gateway::~Gateway() {
  Stop();
  CloseSockets();
}

void Gateway::InitSockets() {
  node_agent_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::router, port_);

  // Create inproc PAIR sockets for self-pipe wakeup pattern
  wakeup_recv_ =
      std::make_shared<zmq::socket_t>(*zmq_context_, zmq::socket_type::pair);
  wakeup_send_ =
      std::make_shared<zmq::socket_t>(*zmq_context_, zmq::socket_type::pair);
  wakeup_recv_->bind("inproc://gateway-wakeup");
  wakeup_send_->connect("inproc://gateway-wakeup");
}

void Gateway::NotifyOutbox() {
  // Send a single byte to wake the Gateway from zmq::poll().
  // Uses dontwait to avoid blocking the caller if the pipe is full.
  // Lock required: called from both Handler and Executor threads,
  // and ZMQ sockets are not thread-safe.
  std::lock_guard<std::mutex> lock(wakeup_mutex_);
  zmq::message_t signal(1);
  static_cast<char*>(signal.data())[0] = 'W';
  [[maybe_unused]] auto result =
      wakeup_send_->send(std::move(signal), zmq::send_flags::dontwait);
}

void Gateway::CloseSockets() {
  if (wakeup_send_) wakeup_send_->close();
  if (wakeup_recv_) wakeup_recv_->close();
  if (node_agent_socket_) {
    node_agent_socket_->close();
  }
}

void Gateway::Start() {
  if (running_.load()) {
    return;
  }
  thread_ = std::thread(SETU_LAUNCH_THREAD([this]() { this->Loop(); },
                                           "CoordinatorGatewayThread"));
}

void Gateway::Stop() {
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
}

void Gateway::Loop() {
  running_ = true;
  while (running_) {
    // Poll for incoming messages from NodeAgents OR wakeup signal
    auto ready =
        Comm::PollForRead({node_agent_socket_, wakeup_recv_}, kPollTimeoutMs);

    for (const auto& socket : ready) {
      if (socket == node_agent_socket_) {
        auto [node_agent_identity, request] =
            Comm::RecvWithIdentity<NodeAgentRequest>(socket);
        auto status =
            inbox_queue_.try_push(InboxMessage{node_agent_identity, request});
        if (status == boost::queue_op_status::closed) {
          return;
        }
      } else if (socket == wakeup_recv_) {
        // Drain all wakeup signals (there may be multiple)
        zmq::message_t drain;
        while (
            wakeup_recv_->recv(drain, zmq::recv_flags::dontwait).has_value()) {
        }
      }
    }

    // Send any outgoing messages (drain all available without blocking)
    try {
      while (!outbox_queue_.empty()) {
        OutboxMessage outbox_msg = outbox_queue_.pull();
        Comm::Send<CoordinatorMessage>(node_agent_socket_,
                                       outbox_msg.node_agent_identity,
                                       outbox_msg.message);
      }
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
