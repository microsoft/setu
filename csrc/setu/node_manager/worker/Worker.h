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
#include "commons/datatypes/Device.h"
#include "commons/enums/Enums.h"
#include "planner/ir/llc/Instruction.h"
#include "planner/ir/ref/RegisterRef.h"
#include "telemetry/MetricsSink.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::CopyOperationId;
using setu::commons::DevicePtr;
using setu::commons::DeviceRank;
using setu::commons::NodeId;
using setu::commons::Queue;
using setu::commons::datatypes::Device;
using setu::commons::enums::ErrorCode;
using setu::planner::ir::llc::Program;
using setu::planner::ir::ref::RegisterRef;
using setu::telemetry::MetricsSinkPtr;
//==============================================================================

/// @brief Work item dispatched to a worker's input queue.
struct WorkerTask {
  CopyOperationId copy_op_id;
  Program program;
  std::chrono::steady_clock::time_point enqueued_at;
};

/// @brief Completion notification pushed by a worker after executing a task.
struct WorkerCompletion {
  CopyOperationId copy_op_id;
  DeviceRank device_rank;
};

//==============================================================================
class Worker {
 public:
  Worker(NodeId node_id, Device device);
  virtual ~Worker();

  /// Bind this worker to an input queue (for receiving tasks) and a
  /// completion queue (for signaling done). Must be called before Start().
  void Bind(Queue<WorkerTask>& input_queue,
            Queue<WorkerCompletion>& completion_queue);

  void Start();
  void Stop();

  [[nodiscard]] bool IsRunning() const { return worker_running_.load(); }
  [[nodiscard]] const Device& GetDevice() const { return device_; }

  virtual void Execute(const Program& program) = 0;
  virtual void Setup() = 0;

  /// Resolve a register reference to its device pointer.
  [[nodiscard]] virtual DevicePtr ResolveRegister(
      const RegisterRef& ref) const = 0;

  /// Set the metrics sink for telemetry submission.
  void SetMetricsSink(MetricsSinkPtr sink);

 protected:
  void WorkerLoop();

  /// @brief Check and report completions for previously dispatched GPU work.
  /// Subclasses override to query CUDA events and push to completion_queue_.
  /// Default: no-op (synchronous workers push completion in WorkerLoop).
  virtual void DrainCompletions() {}

  /// @brief Returns true if there are in-flight programs awaiting completion.
  [[nodiscard]] virtual bool HasPendingCompletions() const { return false; }

  /// @brief Block until there is capacity to dispatch another program.
  /// Called before Execute() to enforce max in-flight limits.
  virtual void WaitForCapacity() {}

  NodeId node_id_;
  Device device_;

  Queue<WorkerTask>* input_queue_ = nullptr;
  Queue<WorkerCompletion>* completion_queue_ = nullptr;

  std::atomic<bool> worker_running_;

  std::thread worker_thread_;

  CopyOperationId current_copy_op_id_;
  MetricsSinkPtr metrics_sink_;
};
//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
