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
#include "commons/Logging.h"
//==============================================================================
namespace setu::commons {
//==============================================================================
void Logger::InitializeLogLevel() {
  const char* env_log_level = std::getenv("SETU_LOG_LEVEL");
  if (!env_log_level) {
    log_level = LogLevel::kInfo;
    return;
  }

  std::string level_str = env_log_level;
  if (level_str == "DEBUG") {
    log_level = LogLevel::kDebug;
  } else if (level_str == "INFO") {
    log_level = LogLevel::kInfo;
  } else if (level_str == "WARNING") {
    log_level = LogLevel::kWarning;
  } else if (level_str == "ERROR") {
    log_level = LogLevel::kError;
  } else if (level_str == "CRITICAL") {
    log_level = LogLevel::kCritical;
  } else {
    log_level = LogLevel::kInfo;
  }
}
//==============================================================================
}  // namespace setu::commons
//==============================================================================
