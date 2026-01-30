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
#include "commons/messages/Messages.h"
#include "commons/utils/SetuCommHelper.h"
#include "commons/utils/ThreadingUtils.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::enums::ErrorCode;
using setu::commons::messages::ExecuteProgramRequest;
using setu::commons::messages::ExecuteProgramResponse;
using setu::commons::utils::SetuCommHelper;
using setu::commons::utils::ZmqHelper;
//==============================================================================
// Worker
//==============================================================================

Worker::Worker(Device device,
               std::size_t reply_port)
    : device_(device),
      reply_port_(reply_port),
      worker_running_(false) {
  InitZmqSockets();
}

Worker::~Worker() {
  Stop();
  CloseZmqSockets();
}

void Worker::Start() {
  if (worker_running_) {
    return;
  }
  worker_running_ = true;
  executor_thread_ =
      std::thread(SETU_LAUNCH_THREAD([this]() { WorkerLoop(); }, "WorkerLoop"));
}

void Worker::Stop() {
  if (!worker_running_) {
    return;
  }
  worker_running_ = false;
  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }
}

void Worker::Setup() {
  // Base implementation is a no-op; NCCLWorker overrides for CUDA/stream setup.
}

void Worker::InitZmqSockets() {
  zmq_context_ = std::make_shared<zmq::context_t>();
  reply_socket_ = ZmqHelper::CreateAndBindSocket(
      zmq_context_, zmq::socket_type::rep, reply_port_);
}

void Worker::CloseZmqSockets() {
  if (reply_socket_) {
    reply_socket_->close();
  }
  if (zmq_context_) {
    zmq_context_->close();
  }
}

void Worker::WorkerLoop() {
  LOG_DEBUG("WorkerLoop started on device {}", device_);

  this->Setup();

  while (worker_running_) {
    auto request = SetuCommHelper::Recv<ExecuteProgramRequest>(reply_socket_);
    this->Execute(request.program);

    ExecuteProgramResponse response(ErrorCode::kSuccess);
    SetuCommHelper::Send(reply_socket_, response);
  }
}

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
