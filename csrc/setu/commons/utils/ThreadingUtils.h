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
#include "commons/Logging.h"
#include "commons/StdCommon.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::Logger;
//==============================================================================
/**
 * @brief Threading utility functions
 */
class ThreadingUtils {
 public:
  /**
   * @brief Launch a thread with a name for logging
   * @param func Thread function to run
   * @param thread_name Name for logging
   * @return Wrapped function with thread name set
   */
  template <typename Func>
  static auto LaunchThread(Func&& func, const std::string& thread_name) {
    return [func = std::forward<Func>(func), thread_name]() {
      // Set thread name in logger for this thread
      Logger::SetThreadName(thread_name);
      func();
    };
  }
};
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
/**
 * @brief Convenience macro for launching threads with names
 */
#define SETU_LAUNCH_THREAD(func, name) \
  setu::commons::utils::ThreadingUtils::LaunchThread(func, name)
//==============================================================================
