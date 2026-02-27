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
using setu::commons::enums::ErrorCode;
using setu::commons::messages::ExecuteRequest;
using setu::commons::messages::OnboardNodeAgentResponse;
using setu::planner::Plan;
//==============================================================================
Executor::Executor(Queue<ExecutorTask>& planner_queue,
                   Queue<OutboxMessage>& outbox_queue, MetaStore& metastore,
                   Planner& planner, OutboxNotifyFn outbox_notify,
                   setu::telemetry::MetricsSinkPtr metrics_sink)
    : planner_queue_(planner_queue),
      outbox_queue_(outbox_queue),
      metastore_(metastore),
      planner_(planner),
      outbox_notify_(std::move(outbox_notify)),
      metrics_sink_(std::move(metrics_sink)) {}

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
      ExecutorTask task = planner_queue_.pull();

      std::visit(
          [&](auto&& alt) {
            using T = std::decay_t<decltype(alt)>;
            if constexpr (std::is_same_v<T, PlannerTask>) {
              HandlePlannerTask(std::move(alt));
            } else if constexpr (std::is_same_v<T, OnboardingTask>) {
              HandleOnboardingTask(std::move(alt));
            }
          },
          std::move(task));
    } catch (const boost::concurrent::sync_queue_is_closed&) {
      return;
    }
  }
}

void Executor::HandlePlannerTask(PlannerTask task) {
  auto t_after_dequeue = std::chrono::steady_clock::now();

  LOG_DEBUG("Executor received task for copy_op_id: {}", task.copy_op_id);

  auto result = planner_.Compile(task.copy_spec, metastore_, task.hints,
                                 task.copy_op_id);
  Plan plan = std::move(result.plan);

  // Submit compilation metrics
  if (metrics_sink_ && metrics_sink_->IsEnabled()) {
    metrics_sink_->Submit(
        setu::telemetry::MetricsMessage{std::move(result.metrics)});
  }

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
  LOG_INFO("Executor: copy_op_id={}, total={}us", task.copy_op_id,
           to_us(t_end - t_after_dequeue));
}

void Executor::HandleOnboardingTask(OnboardingTask task) {
  LOG_INFO("Executor processing OnboardingTask ({} devices)",
           task.register_sets.size());
  planner_.AddBackendRegisterSets(task.register_sets);

  OnboardNodeAgentResponse response(task.request_id, ErrorCode::kSuccess);
  PushOutbox(OutboxMessage{task.node_agent_identity, response});
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
