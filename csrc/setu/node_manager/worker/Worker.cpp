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
#include "commons/utils/ThreadingUtils.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
Worker::Worker(NodeId node_id, Device device)
    : node_id_(node_id), device_(device), worker_running_{false} {}

Worker::~Worker() { Stop(); }

void Worker::Bind(Queue<WorkerTask>& input_queue,
                  Queue<WorkerCompletion>& completion_queue) {
  input_queue_ = &input_queue;
  completion_queue_ = &completion_queue;
}

void Worker::Start() {
  if (worker_running_.load()) return;

  ASSERT_VALID_RUNTIME(input_queue_ != nullptr,
                       "Worker must be bound to an input queue before Start()");
  ASSERT_VALID_RUNTIME(
      completion_queue_ != nullptr,
      "Worker must be bound to a completion queue before Start()");

  worker_running_ = true;
  worker_thread_ = std::thread(
      SETU_LAUNCH_THREAD([this]() { WorkerLoop(); }, "WorkerLoop"));
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

void Worker::SetMetricsSink(MetricsSinkPtr sink) {
  metrics_sink_ = std::move(sink);
}

void Worker::WorkerLoop() {
  LOG_DEBUG("WorkerLoop started on device {}", device_);

  this->Setup();
  while (worker_running_) {
    try {
      auto task = input_queue_->pull();
      current_copy_op_id_ = task.copy_op_id;

      auto t0 = std::chrono::steady_clock::now();

      this->Execute(task.program);

      auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
      LOG_DEBUG("Worker[{}]: Execute took {}us, {} instructions", device_, dt,
                task.program.size());

      completion_queue_->push(
          WorkerCompletion{task.copy_op_id, device_.LocalDeviceIndex()});
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}
//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
