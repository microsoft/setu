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

  auto to_us = [](auto d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  };

  this->Setup();
  while (worker_running_) {
    try {
      auto t_loop_start = std::chrono::steady_clock::now();

      // Drain completions for previously dispatched GPU work
      DrainCompletions();
      auto t_after_drain = std::chrono::steady_clock::now();

      // Ensure we have capacity for another program
      WaitForCapacity();
      auto t_after_capacity = std::chrono::steady_clock::now();

      // Pull next task. If we have pending in-flight work, use non-blocking
      // pull so we keep checking completions. Otherwise block.
      WorkerTask task;
      bool got_task_immediately = false;
      if (HasPendingCompletions()) {
        auto status = input_queue_->try_pull(task);
        if (status != boost::concurrent::queue_op_status::success) {
          LOG_DEBUG("PIPELINE[{}]: starved, drain={}us capacity={}us pending={}",
                    device_,
                    to_us(t_after_drain - t_loop_start),
                    to_us(t_after_capacity - t_after_drain),
                    HasPendingCompletions());
          continue;
        }
        got_task_immediately = true;
      } else {
        task = input_queue_->pull();
      }
      auto t_after_pull = std::chrono::steady_clock::now();

      current_copy_op_id_ = task.copy_op_id;

      auto queue_latency_us = to_us(t_after_pull - task.enqueued_at);
      auto queue_depth = input_queue_->size();

      this->Execute(task.program);
      auto t_after_execute = std::chrono::steady_clock::now();

      LOG_DEBUG("PIPELINE[{}]: copy_op={} drain={}us capacity_wait={}us "
                "pull_wait={}us dispatch={}us queue_latency={}us "
                "queue_depth={} queue_had_task={}",
                device_, current_copy_op_id_,
                to_us(t_after_drain - t_loop_start),
                to_us(t_after_capacity - t_after_drain),
                to_us(t_after_pull - t_after_capacity),
                to_us(t_after_execute - t_after_pull),
                queue_latency_us,
                queue_depth,
                got_task_immediately);

    } catch (const boost::concurrent::sync_queue_is_closed&) {
      // Queue closed — drain any remaining in-flight completions before exit
      while (HasPendingCompletions()) {
        DrainCompletions();
        if (HasPendingCompletions()) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
      }
      return;
    }
  }
}
//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
