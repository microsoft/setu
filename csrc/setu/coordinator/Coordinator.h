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
#include "commons/BoostCommon.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
//==============================================================================
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorShardMetadata.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "coordinator/Executor.h"
#include "coordinator/Gateway.h"
#include "coordinator/Handler.h"
#include "coordinator/Types.h"
#include "metastore/MetaStore.h"
#include "planner/Planner.h"
#include "telemetry/MetricsSink.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::Queue;
using setu::commons::datatypes::TensorShardMetadata;
using setu::commons::datatypes::TensorShardSpec;
using setu::metastore::MetaStore;
using setu::planner::PlannerPtr;
//==============================================================================

/// @brief Coordinator orchestrates copy operations between NodeAgents.
///
///   NodeAgents <---> [Gateway] <---> inbox_queue_  ---> Handler
///                        ^
///                        |
///                    outbox_queue_ <--- Handler / Executor
///
/// Gateway owns the ZMQ socket. Handler and Executor are pure business logic
/// communicating through thread-safe queues.
class Coordinator {
 public:
  Coordinator(std::size_t port, PlannerPtr planner,
              std::string metrics_endpoint = "");
  ~Coordinator();

  std::optional<TensorShardMetadata> RegisterTensorShard(
      const TensorShardSpec& shard_spec);

  std::optional<CopyOperationId> SubmitCopy(const CopySpec& copy_spec);

  void PlanExecuted(CopyOperationId copy_op_id);

  void Start();
  void Stop();

 private:
  std::size_t port_;
  std::string metrics_endpoint_;

  std::shared_ptr<zmq::context_t> zmq_context_;

  MetaStore metastore_;
  PlannerPtr planner_;

  // Internal message queues
  Queue<InboxMessage> inbox_queue_;
  Queue<OutboxMessage> outbox_queue_;

  /// Queue of PlannerTasks (CopyOperationId + CopySpec) for the Executor to
  /// compile and dispatch
  Queue<PlannerTask> planner_queue_;

  std::unique_ptr<Gateway> gateway_;
  std::unique_ptr<Handler> handler_;
  std::unique_ptr<Executor> executor_;
};
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
