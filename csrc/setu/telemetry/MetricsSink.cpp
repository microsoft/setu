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
#include "telemetry/MetricsSink.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::telemetry {
//==============================================================================

MetricsSink::MetricsSink(ZmqContextPtr zmq_context, std::string server_endpoint)
    : endpoint_(std::move(server_endpoint)) {
  ASSERT_VALID_POINTER_ARGUMENT(zmq_context);

  socket_ =
      std::make_shared<zmq::socket_t>(*zmq_context, zmq::socket_type::push);
  socket_->set(zmq::sockopt::linger, 0);
  socket_->set(zmq::sockopt::sndhwm, 1000);
  socket_->connect(endpoint_);

  LOG_DEBUG("MetricsSink connected to {}", endpoint_);
}

void MetricsSink::Submit(const MetricsMessage& message) {
  if (!enabled_.load(std::memory_order_relaxed)) {
    return;
  }

  // Serialize the variant using BinaryWriter
  BinaryBuffer buffer;
  BinaryWriter writer(buffer);
  writer.Write(message);

  zmq::message_t zmq_msg(buffer.data(), buffer.size());

  // Non-blocking send: silently drop if server is unavailable
  auto result = socket_->send(std::move(zmq_msg), zmq::send_flags::dontwait);
  if (!result.has_value()) {
    LOG_DEBUG("MetricsSink: message dropped (server unavailable)");
  }
}

void MetricsSink::SetEnabled(bool enabled) {
  enabled_.store(enabled, std::memory_order_relaxed);
}

bool MetricsSink::IsEnabled() const {
  return enabled_.load(std::memory_order_relaxed);
}

//==============================================================================
}  // namespace setu::telemetry
//==============================================================================
