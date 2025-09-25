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
#include "commons/Formatter.h"
#include "commons/LogLevels.h"
#include "commons/StdCommon.h"
//==============================================================================
// Compile-time function to extract filename from path
constexpr const char* ExtractFilename(const char* path) {
  const char* filename = path;
  for (const char* p = path; *p != '\0'; ++p) {
    if (*p == '/' || *p == '\\') {
      filename = p + 1;
    }
  }
  return filename;
}
//==============================================================================
#define LOG_DEBUG(...)                                        \
  setu::commons::Logger::log(setu::commons::LogLevel::kDebug, \
                             ExtractFilename(__FILE__), __LINE__, __VA_ARGS__)
#define LOG_INFO(...)                                        \
  setu::commons::Logger::log(setu::commons::LogLevel::kInfo, \
                             ExtractFilename(__FILE__), __LINE__, __VA_ARGS__)
#define LOG_WARNING(...)                                        \
  setu::commons::Logger::log(setu::commons::LogLevel::kWarning, \
                             ExtractFilename(__FILE__), __LINE__, __VA_ARGS__)
#define LOG_ERROR(...)                                        \
  setu::commons::Logger::log(setu::commons::LogLevel::kError, \
                             ExtractFilename(__FILE__), __LINE__, __VA_ARGS__)
#define LOG_CRITICAL(...)                                        \
  setu::commons::Logger::log(setu::commons::LogLevel::kCritical, \
                             ExtractFilename(__FILE__), __LINE__, __VA_ARGS__)
//==============================================================================
#define RAISE_ERROR(error_type, error_message, format_str, ...)             \
  do {                                                                      \
    std::string message = std::format(format_str, ##__VA_ARGS__);           \
    std::string file = ExtractFilename(__FILE__);                           \
    std::string line = std::to_string(__LINE__);                            \
    std::string backtrace =                                                 \
        boost::stacktrace::to_string(boost::stacktrace::stacktrace(0, 10)); \
    std::string full_message =                                              \
        std::format("{}: Message: {} File: {}:{} \nBacktrace:\n{}",         \
                    error_message, message, file, line, backtrace);         \
    LOG_CRITICAL("{}", full_message);                                       \
    throw error_type(full_message);                                         \
  } while (0)
//==============================================================================
#define RAISE_INVALID_ARGUMENTS_ERROR(format_str, ...)                \
  RAISE_ERROR(std::invalid_argument, "Invalid arguments", format_str, \
              ##__VA_ARGS__)
//==============================================================================
#define ASSERT_VALID_ARGUMENTS(x, format_str, ...)                     \
  if (!(x)) {                                                          \
    RAISE_ERROR(std::invalid_argument, "Invalid argument", format_str, \
                ##__VA_ARGS__);                                        \
  }
//==============================================================================
#define ASSERT_VALID_POINTER_ARGUMENT(x)                           \
  if (x == nullptr) {                                              \
    RAISE_ERROR(std::invalid_argument, "Invalid pointer argument", \
                "{} is nullptr", #x);                              \
  }
//==============================================================================
#define ASSERT_VALID_RUNTIME(x, format_str, ...)                 \
  if (!(x)) {                                                    \
    RAISE_ERROR(std::runtime_error, "Runtime error", format_str, \
                ##__VA_ARGS__);                                  \
  }
//==============================================================================
#define RAISE_RUNTIME_ERROR(format_str, ...) \
  RAISE_ERROR(std::runtime_error, "Runtime error", format_str, ##__VA_ARGS__)
//==============================================================================
namespace setu::commons {
//==============================================================================
enum class ProcessType { MAIN = 0, CONTROLLER = 1, WORKER = 2 };

constexpr const char* ProcessTypeToString(ProcessType type) {
  switch (type) {
    case ProcessType::MAIN:
      return "M";
    case ProcessType::CONTROLLER:
      return "C";
    case ProcessType::WORKER:
      return "W";
    default:
      return "U";  // Unknown
  }
}
//==============================================================================
constexpr const char* LogLevelToString(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "DEBUG";
    case LogLevel::kInfo:
      return "INFO";
    case LogLevel::kWarning:
      return "WARNING";
    case LogLevel::kError:
      return "ERROR";
    case LogLevel::kCritical:
      return "CRITICAL";
    default:
      return "UNKNOWN";
  }
}
//==============================================================================
/**
 * @brief Logging utility for Setu
 *
 * Provides centralized logging functionality with log level control.
 */
struct Logger {
 public:
  static void InitializeLogLevel();
  //==============================================================================
  template <typename... Args>
  static inline void log(LogLevel severity, const char* file, int line,
                         std::format_string<Args...> format, Args&&... args) {
    if (severity < log_level) {
      return;
    }

    // Cache process and thread info
    static thread_local pid_t cached_pid = getpid();
    static thread_local std::thread::id cached_tid = std::this_thread::get_id();
    static thread_local std::string tid_str = [&]() {
      std::ostringstream oss;
      oss << cached_tid;
      return oss.str();
    }();

    std::string message = std::format(format, std::forward<Args>(args)...);
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::system_clock::now().time_since_epoch()) %
              1000000;

    std::fprintf(
        stderr,
        "%02d%02d%02d %02d:%02d:%02d.%06ld %d:%s %s:%s] [%s] [%s:%d] %s\n",
        tm.tm_year % 100, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min,
        tm.tm_sec, ms.count(), cached_pid, tid_str.c_str(),
        process_identifier_.c_str(), GetThreadName().c_str(),
        LogLevelToString(severity), file, line, message.c_str());
  }
  //==============================================================================
  static void SetRank(int rank) {
    rank_ = rank;
    UpdateProcessIdentifier();
  }
  static void SetReplicaId(int replica_id) {
    replica_id_ = replica_id;
    UpdateProcessIdentifier();
  }
  static void SetProcessType(ProcessType type) {
    process_type_ = type;
    UpdateProcessIdentifier();
  }
  static void SetThreadName(const std::string& name) { GetThreadName() = name; }
  [[nodiscard]] static int GetRank() { return rank_; }
  [[nodiscard]] static int GetReplicaId() { return replica_id_; }
  [[nodiscard]] static ProcessType GetProcessType() { return process_type_; }
  static const std::string& GetProcessIdentifier() {
    return process_identifier_;
  }
  static std::string& GetThreadName() {
    static thread_local std::string thread_name = "main";
    return thread_name;
  }

 private:
  static void UpdateProcessIdentifier() {
    switch (process_type_) {
      case ProcessType::MAIN:
        process_identifier_ = "M";
        break;
      case ProcessType::CONTROLLER:
        process_identifier_ = std::format("C-{}", replica_id_);
        break;
      case ProcessType::WORKER:
        process_identifier_ = std::format("W-{}-{}", replica_id_, rank_);
        break;
      default:
        process_identifier_ = "U";
        break;
    }
  }

 public:
  //==============================================================================
  static inline LogLevel log_level;
  static inline int rank_ = -1;
  static inline int replica_id_ = -1;
  static inline ProcessType process_type_ = ProcessType::MAIN;
  static inline std::string process_identifier_ = "M";
};
//==============================================================================
}  // namespace setu::commons
//==============================================================================
