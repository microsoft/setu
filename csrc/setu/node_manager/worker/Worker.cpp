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
#include "node_manager/worker/Worker.h"
//==============================================================================
#include "commons/Logging.h"
#include "messaging/Messages.h"
#include "commons/utils/Comm.h"
#include "commons/utils/ThreadingUtils.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::RequestId;
using setu::commons::enums::ErrorCode;
using setu::commons::messages::ExecuteProgramRequest;
using setu::commons::messages::ExecuteProgramResponse;
using setu::commons::messages::SubmitCopyResponse;
using setu::commons::messages::WaitForCopyResponse;
using setu::commons::utils::Comm;
using setu::commons::utils::ZmqHelper;
using setu::ir::Instruction;
//==============================================================================
Worker::Worker(NodeId node_id, Device device, ZmqContextPtr zmq_context,
               std::string inproc_endpoint)
    : node_id_(node_id),
      device_(device),
      inproc_endpoint_(std::move(inproc_endpoint)),
      zmq_context_(std::move(zmq_context)),
      worker_running_{false} {
  ASSERT_VALID_POINTER_ARGUMENT(zmq_context_);
  InitZmqSockets();
}

Worker::~Worker() {
  Stop();
  CloseZmqSockets();
}

void Worker::Start() {
  if (worker_running_) return;

  LOG_DEBUG("Starting Worker");
  if (!worker_running_.load()) {
    worker_running_ = true;
    worker_thread_ = std::thread(
        SETU_LAUNCH_THREAD([this]() { WorkerLoop(); }, "WorkerLoop"));
  }
}

void Worker::Stop() {
  if (!worker_running_) {
    return;
  }
  worker_running_ = false;
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

void Worker::InitZmqSockets() {
  LOG_DEBUG("Initializing ZMQ sockets for endpoint {}", inproc_endpoint_);

  socket_ = ZmqHelper::CreateAndBindInprocSocket(
      zmq_context_, zmq::socket_type::rep, inproc_endpoint_);

  LOG_DEBUG("Worker: Initialized ZMQ socket on {}", endpoint_);
}

void Worker::CloseZmqSockets() {
  LOG_DEBUG("Worker: Closing ZMQ socket on {}", endpoint_);

  if (socket_) socket_->close();
  // Note: zmq_context_ is shared and not owned by Worker, so don't close it

  LOG_DEBUG("Worker: Closed ZMQ socket on {}", endpoint_);
}

void Worker::WorkerLoop() {
  LOG_DEBUG("WorkerLoop started on device {}", device_);

  this->Setup();
  while (worker_running_) {
    // Receive ExecuteProgramRequest from NodeAgent
    auto request = Comm::Recv<ExecuteProgramRequest>(socket_);
    const auto& program = request.program;

    LOG_DEBUG("Worker received program with {} instructions", program.size());

    // Execute each instruction in the program
    this->Execute(program);

    LOG_DEBUG("Worker completed executing all instructions");

    // Send acknowledgment back to NodeAgent
    ExecuteProgramResponse response(RequestId{}, ErrorCode::kSuccess);
    Comm::Send(socket_, response);
  }
}
//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
