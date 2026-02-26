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
#include "coordinator/Executor.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/QueueUtils.h"
#include "commons/utils/ThreadingUtils.h"
#include "messaging/Messages.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::messages::ExecuteRequest;
using setu::planner::Plan;
//==============================================================================
Executor::Executor(Queue<PlannerTask>& planner_queue,
                   Queue<OutboxMessage>& outbox_queue, MetaStore& metastore,
                   Planner& planner, OutboxNotifyFn outbox_notify)
    : planner_queue_(planner_queue),
      outbox_queue_(outbox_queue),
      metastore_(metastore),
      planner_(planner),
      outbox_notify_(std::move(outbox_notify)) {}

void Executor::PushOutbox(OutboxMessage msg) {
  outbox_queue_.push(std::move(msg));
  outbox_notify_();
}

void Executor::Start() {
  if (running_.load()) {
    return;
  }
  thread_ = std::thread(SETU_LAUNCH_THREAD([this]() { this->Loop(); },
                                           "CoordinatorExecutorThread"));
}

void Executor::Stop() {
  running_ = false;

  if (thread_.joinable()) {
    thread_.join();
  }
}

void Executor::Loop() {
  running_ = true;
  while (running_) {
    try {
      PlannerTask task = planner_queue_.pull();
      auto t_after_dequeue = std::chrono::steady_clock::now();

      LOG_DEBUG("Executor received task for copy_op_id: {}", task.copy_op_id);

      auto t_compile_start = std::chrono::steady_clock::now();
      Plan plan = planner_.Compile(task.copy_spec, metastore_, task.hints);
      auto t_compile_end = std::chrono::steady_clock::now();

      LOG_DEBUG("Compiled plan:\n{}", plan);

      // Fragment the plan to into per-node fragments
      auto fragments = plan.Fragments();

      // Send ExecuteRequest to each node agent
      for (auto& [node_id, node_plan] : fragments) {
        Identity node_identity = boost::uuids::to_string(node_id) + "_dealer";

        ExecuteRequest execute_request(task.copy_op_id, std::move(node_plan));

        PushOutbox(OutboxMessage{node_identity, execute_request});
      }

      // Set expected responses
      // memory order release so Handler thread can pick it up (using memory
      // order aqcuire)
      task.state->expected_responses.store(fragments.size(),
                                           std::memory_order_release);

      auto t_end = std::chrono::steady_clock::now();
      auto to_us = [](auto d) {
        return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
      };
      LOG_INFO(
          "Executor: copy_op_id={}, compile={}us, fragment+dispatch={}us, "
          "total={}us",
          task.copy_op_id, to_us(t_compile_end - t_compile_start),
          to_us(t_end - t_compile_end), to_us(t_end - t_after_dequeue));

    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
