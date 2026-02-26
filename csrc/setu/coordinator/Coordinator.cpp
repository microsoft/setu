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
#include "coordinator/Coordinator.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::coordinator {
//==============================================================================
using setu::commons::datatypes::CopySpec;
//==============================================================================
Coordinator::Coordinator(std::size_t port, PlannerPtr planner,
                         std::string metrics_endpoint)
    : port_(port),
      metrics_endpoint_(std::move(metrics_endpoint)),
      zmq_context_(std::make_shared<zmq::context_t>()),
      planner_(planner) {
  // Create MetricsSink instances if metrics_endpoint is configured.
  // Each thread gets its own sink (ZMQ sockets are not thread-safe).
  setu::telemetry::MetricsSinkPtr handler_sink;
  setu::telemetry::MetricsSinkPtr executor_sink;
  if (!metrics_endpoint_.empty()) {
    handler_sink = std::make_shared<setu::telemetry::MetricsSink>(
        zmq_context_, metrics_endpoint_);
    executor_sink = std::make_shared<setu::telemetry::MetricsSink>(
        zmq_context_, metrics_endpoint_);
  }

  gateway_ = std::make_unique<Gateway>(zmq_context_, port_, inbox_queue_,
                                       outbox_queue_);

  auto outbox_notify = [this]() { gateway_->NotifyOutbox(); };

  handler_ =
      std::make_unique<Handler>(inbox_queue_, outbox_queue_, metastore_,
                                planner_queue_, outbox_notify, handler_sink);
  executor_ =
      std::make_unique<Executor>(planner_queue_, outbox_queue_, metastore_,
                                 *planner_, outbox_notify, executor_sink);
}

Coordinator::~Coordinator() {
  Stop();
  if (zmq_context_) {
    zmq_context_->close();
  }
}

void Coordinator::Start() {
  LOG_DEBUG("Starting Coordinator");
  gateway_->Start();
  handler_->Start();
  executor_->Start();
}

void Coordinator::Stop() {
  LOG_DEBUG("Stopping Coordinator");

  inbox_queue_.close();
  planner_queue_.close();
  outbox_queue_.close();

  gateway_->Stop();
  handler_->Stop();
  executor_->Stop();
}

std::optional<TensorShardMetadata> Coordinator::RegisterTensorShard(
    const TensorShardSpec& shard_spec) {
  LOG_DEBUG("Registering tensor shard: {}", shard_spec.name);

  // TODO: Implement tensor shard registration
  return std::nullopt;
}

std::optional<CopyOperationId> Coordinator::SubmitCopy(
    const CopySpec& copy_spec) {
  LOG_DEBUG("Submitting copy operation from {} to {}", copy_spec.src_name,
            copy_spec.dst_name);

  // TODO: Implement copy submission and plan generation
  return std::nullopt;
}

void Coordinator::PlanExecuted(CopyOperationId copy_op_id) {
  LOG_DEBUG("Plan executed for copy operation ID: {}", copy_op_id);

  // TODO: Implement plan execution completion handling
}
//==============================================================================
}  // namespace setu::coordinator
//==============================================================================
