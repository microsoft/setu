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
#include "coordinator/Types.h"
#include "metastore/MetaStore.h"
#include "planner/Planner.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::metastore::MetaStore;
using setu::planner::Planner;
//==============================================================================

/// @brief Compiles CopySpecs into execution plans and dispatches them to
/// NodeAgents.
///
/// Executor runs on a dedicated thread, pulling PlannerTasks from the planner
/// queue. For each task it compiles a Plan, fragments it per-node, and sends
/// ExecuteRequests through the outbox queue.
class Executor {
 public:
  Executor(Queue<PlannerTask>& planner_queue,
           Queue<OutboxMessage>& outbox_queue, MetaStore& metastore,
           Planner& planner, OutboxNotifyFn outbox_notify);

  void Start();
  void Stop();

 private:
  void Loop();

  void PushOutbox(OutboxMessage msg);

  Queue<PlannerTask>& planner_queue_;
  Queue<OutboxMessage>& outbox_queue_;
  MetaStore& metastore_;
  Planner& planner_;
  OutboxNotifyFn outbox_notify_;

  std::thread thread_;
  std::atomic<bool> running_{false};
};
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
