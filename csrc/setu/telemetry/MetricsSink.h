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
#include "commons/utils/ZmqHelper.h"
#include "telemetry/MetricsData.h"
#include "telemetry/NCCLWorkerMetrics.h"
//==============================================================================
namespace setu::telemetry {
//==============================================================================
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;
//==============================================================================

/// @brief Non-blocking fire-and-forget metrics submitter over ZMQ PUSH.
///
/// Each thread that needs to submit metrics should create its own
/// MetricsSink instance (ZMQ sockets are not thread-safe).
/// If the server is down, messages are silently dropped.
class MetricsSink {
 public:
  MetricsSink(ZmqContextPtr zmq_context, std::string server_endpoint);

  /// @brief Serialize and send a MetricsMessage. Non-blocking.
  void Submit(const MetricsMessage& message);

  void SetEnabled(bool enabled);
  [[nodiscard]] bool IsEnabled() const;

 private:
  ZmqSocketPtr socket_;
  std::string endpoint_;
  std::atomic<bool> enabled_{true};
};

using MetricsSinkPtr = std::shared_ptr<MetricsSink>;

//==============================================================================
}  // namespace setu::telemetry
//==============================================================================
