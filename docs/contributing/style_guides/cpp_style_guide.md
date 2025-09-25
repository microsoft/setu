# Setu C++ Style Guide

## Introduction

This style guide outlines the coding conventions and best practices for writing C++ code within the Setu project. It follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with Setu-specific modifications and leverages modern C++20/23 features extensively. All code should adhere to these guidelines to ensure consistency, readability, maintainability, and effective collaboration.

## Key Goals

- **Readability**: Code should be easy to understand and follow by any developer
- **Consistency**: Uniformity in style across the project reduces cognitive load and improves predictability
- **Maintainability**: Well-structured and clearly written code is easier to modify, debug, and extend over time
- **Collaboration**: Shared style guidelines facilitate seamless teamwork and code integration
- **Modern C++**: Embrace C++20/23 features for type safety, performance, and expressiveness

## File Organization

### Header File Structure

All header files must follow this exact structure:

```cpp
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
#include "commons/StdCommon.h"      // Standard C++ headers
#include "commons/TorchCommon.h"    // PyTorch integration (if needed)
#include "commons/BoostCommon.h"    // Boost utilities (if needed)
//==============================================================================
#include "commons/ClassTraits.h"
#include "commons/Logging.h"
#include "project_headers.h"
//==============================================================================
namespace setu::commons {  // Use nested namespace syntax (C++17)
//==============================================================================

// Class declarations and implementations

//==============================================================================
}  // namespace setu::commons
//==============================================================================
```

### Common Header Files

Setu uses centralized header files to ensure consistent imports and reduce compilation time:

**StdCommon.h** - All standard C++ headers:
- Containers: `<memory>`, `<vector>`, `<unordered_map>`, `<array>`
- Algorithms: `<algorithm>`, `<numeric>`, `<functional>`
- Concurrency: `<thread>`, `<mutex>`, `<atomic>`, `<future>`
- Utilities: `<optional>`, `<variant>`, `<string_view>`, `<format>`

**TorchCommon.h** - PyTorch integration:
- Core PyTorch: `<torch/all.h>`, `<torch/extension.h>`
- CUDA integration: `<torch/cuda.h>`, `<c10/cuda/CUDAGuard.h>`
- Distributed: `<torch/csrc/distributed/c10d/ProcessGroup.hpp>`

**BoostCommon.h** - Boost utilities:
- Concurrent queues: `boost::concurrent::sync_queue`, `boost::concurrent::sync_priority_queue`
- Type aliases: `Queue<T>`, `PriorityQueue<T>`

### Implementation File Structure

```cpp
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
#include "corresponding_header.h"

#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"  // If needed
#include "other_project_headers.h"
//==============================================================================
namespace setu::commons {
//==============================================================================
// Using declarations - sorted alphabetically
using setu::commons::NonCopyable;
using setu::commons::datatypes::SessionPtr;
using setu::commons::datatypes::TokenId;
//==============================================================================

// Implementation code

//==============================================================================
}  // namespace setu::commons
//==============================================================================
```

## Namespace Conventions

### Namespace Organization

Setu uses a hierarchical namespace structure that mirrors the directory organization:

```cpp
// Directory: csrc/include/setu/native/core/scheduler/
namespace setu::commons::controller {
    // Code for scheduler components
}

// Directory: csrc/include/setu/native/llm/worker/model_executor/layers/
namespace setu::commons::llm::worker::model_executor::layers {
    // Code for model layers
}
```

### Namespace Guidelines

1. **Use C++17 nested namespace syntax**:
```cpp
// ✅ Good
namespace setu::commons::controller {

// ❌ Don't do this
namespace setu {
namespace native {
namespace core {
namespace scheduler {
```

2. **Place using declarations after namespace declaration**:
```cpp
namespace setu::commons::controller {
//==============================================================================
// Sort using declarations alphabetically
using setu::commons::NonCopyable;
using setu::commons::PageTable;
using setu::commons::datatypes::SessionPtr;
//==============================================================================
```

3. **Use namespace aliases for deeply nested namespaces**:
```cpp
namespace setu::commons {
//==============================================================================
using setu::commons::time_utils::now_s;
//==============================================================================
```

4. **Never use `using namespace` in headers**:
```cpp
// ❌ Never do this in headers
using namespace std;
using namespace setu::commons::datatypes;

// ✅ Use specific using declarations instead
using setu::commons::datatypes::SessionPtr;
using setu::commons::datatypes::TokenId;
```

### Benefits of Common Headers

This approach provides:
- **Consistency**: Uniform header inclusion across all files
- **Dependency Management**: Clear separation of external library dependencies
- **Maintenance**: Centralized management of library versions and compatibility

## Naming Conventions

### Classes, Structs, Types, and Enums
- **PascalCase**: `LogicalTokenPage`, `SamplingParams`, `SessionStatus`
- **Abstract Base Classes**: Use `Abstract` prefix, not `Base`: `AbstractReplicaController`, `AbstractSessionBatcher`
- **Enum Classes**: Always use `enum class` with `kPascalCase` values

```cpp
enum class SessionStatus {
  kWaiting,
  kRunning, 
  kFinished,
  kFinishedEOS,
  kFinishedMaxTokens
};
```

### Functions and Methods
- **PascalCase**: `ToString()`, `GetStatus()`, `SetStatus()`, `Forward()`
- **Getters/Setters**: Prefix with "Get"/"Set"

```cpp
[[nodiscard]] std::string ToString() const;
[[nodiscard]] SessionStatus GetStatus() const;
void SetStatus(SessionStatus status /*[in]*/);
```

### Variables
- **snake_case**: `prompt_token_ids`, `arrival_time`, `hidden_states`
- **Member variables**: snake_case with trailing underscore: `state_`, `tokens_`
- **Boolean variables**: Use descriptive predicates: `is_finished_`, `has_value_`

```cpp
class Session {
private:
    SessionStatus state_;
    TokenIdsPtr prompt_token_ids_;
    bool prefill_finished_;
};
```

### Constants
- **PascalCase with k prefix**: `kSamplingEps`

```cpp
constexpr double kSamplingEps = 1e-5;
```

### Type Aliases and Type System

#### Type Alias Conventions
- **PascalCase** with descriptive suffixes:
- **Always define type aliases for IDs and domain concepts**:

```cpp
// IDs and domain types
using SessionId = std::string;
using TokenId = std::int32_t;
using PageId = std::int32_t;
using ReplicaId = std::int32_t;
using Rank = std::int32_t;
using LayerId = std::int32_t;
using TimeS = double;  // Time in seconds

// Container types
using TokenIds = std::vector<TokenId>;
using TokenIdsPtr = std::shared_ptr<TokenIds>;
using PageTable = std::vector<PageId>;
using PageTablePtr = std::shared_ptr<PageTable>;

// Smart pointer types
using SessionPtr = std::shared_ptr<const Session>;
using Sessions = Sessions;
```

#### Fixed-Width Integer Types
- **Always use fixed-width integer types from `<cstdint>`**:
- **Avoid using `int`, `long`, etc.**:

```cpp
// ✅ Good - explicit size
std::int32_t num_tokens;
std::uint64_t memory_size;
std::size_t container_size;

// ❌ Don't do this - ambiguous size
int count;
long offset;
unsigned int flags;
```

#### Unsigned vs Signed Types
- **Use unsigned types for non-negative quantities**:
- **Use signed types only when negative values are meaningful**:

```cpp
// ✅ Good - quantities that can't be negative
std::size_t session_length;
std::uint32_t num_layers;
std::uint64_t token_count;

// ✅ Good - values that can be negative
std::int32_t temperature_delta;  // Can be positive or negative
std::int64_t time_offset_ms;     // Can represent past or future

// ❌ Don't do this
std::int32_t session_length;  // Length can't be negative
std::int32_t num_tokens;       // Count can't be negative
```

#### Avoid std::pair and std::tuple
- **Create named structs instead of using pairs/tuples**:
- **Rule of thumb: If you use the type 3+ times, make it a struct**:

```cpp
// ❌ Don't do this
std::pair<std::int32_t, std::string> token_info;
std::tuple<SessionId, TokenIds, SamplingParams> session_data;
std::vector<std::pair<PageId, std::size_t>> page_usage;

// ✅ Do this instead
struct TokenInfo {
    TokenId id;
    std::string text;
};

struct SessionData {
    SessionId session_id;
    TokenIds tokens;
    SamplingParams params;
};

struct PageUsage {
    PageId page_id;
    std::size_t num_tokens;
};
```

## Class Design and Inheritance

### Base Class Traits

Use the standard Setu class traits from `setu/commons/ClassTraits.h`:

```cpp
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/ClassTraits.h"
//==============================================================================

// For classes that should not be copied
class SessionManager : public NonCopyable {
    // Implementation
};

// For classes that should not be moved or copied
class AbstractModelRunner : public NonCopyableNonMovable {
    // Implementation  
};

// For utility classes with only static methods
class ModelUtils : public StaticClass {
public:
    static bool ValidateConfig(const ModelConfig& config);
};
```

### Constructor Patterns

- Use member initializer lists
- Validate all pointer parameters immediately
- Place each initialization on its own line for readability

```cpp
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/Logging.h"
//==============================================================================

Session::Session(
    const std::string session_id_param,
    const std::string prompt_param, 
    const TokenIdsPtr prompt_token_ids_param,
    const std::size_t page_size_param,
    const TokenId eos_token_id_param,
    const TimeS arrival_time_param,
    const SamplingParams sampling_params_param)
    : session_id(session_id_param),
      prompt(prompt_param),
      prompt_token_ids(prompt_token_ids_param),
      page_size(page_size_param),
      eos_token_id(eos_token_id_param),
      arrival_time(arrival_time_param),
      sampling_params(sampling_params_param),
      state_(SessionStatus::Waiting),
      prefill_finished_(false) {
  
  ASSERT_VALID_POINTER_ARGUMENT(prompt_token_ids);
  ASSERT_VALID_ARGUMENTS(page_size > 0, "Page size must be positive");
  ASSERT_VALID_ARGUMENTS(!prompt.empty(), "Prompt cannot be empty");
}
```

## Validation and Error Handling

### Assertion Macros

Always use Setu validation macros from `setu/commons/Logging.h`:

```cpp
// For null pointer validation
ASSERT_VALID_POINTER_ARGUMENT(ptr);

// For runtime conditions with formatted messages
ASSERT_VALID_RUNTIME(condition, "Failed because: {}", reason);

// For argument validation with formatted messages  
ASSERT_VALID_ARGUMENTS(value > 0, "Value {} must be positive", value);
```

### Error Throwing Macros

```cpp
// For runtime errors
RAISE_RUNTIME_ERROR("Operation failed: {}", error_message);

// For invalid arguments
RAISE_INVALID_ARGUMENTS_ERROR("Invalid parameter: {}", param_name);
```

### Example Usage

```cpp
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/Logging.h"
//==============================================================================

void ProcessSession(const SessionPtr& session /*[in]*/) {
    ASSERT_VALID_POINTER_ARGUMENT(session);
    ASSERT_VALID_RUNTIME(session->IsRunning(), 
        "Session {} is not in running state", session->session_id);
    
    if (session->GetOutputLength() > MAX_SEQUENCE_LENGTH) {
        RAISE_INVALID_ARGUMENTS_ERROR(
            "Session {} exceeds maximum length {}", 
            session->session_id, MAX_SEQUENCE_LENGTH);
    }
    
    // Process session...
}
```

## Logging

Use the structured logging macros from `setu/commons/Logging.h`:

```cpp
LOG_DEBUG("Debug info: value = {}", value);
LOG_INFO("Processing {} sessions", num_sessions);
LOG_WARNING("Memory usage is high: {}MB", memory_mb);
LOG_ERROR("Failed to allocate memory for session {}", session_id);
LOG_CRITICAL("System is in unrecoverable state");
```
Never use **std::cout**, always use the logging macros instead of direct output:

```cpp
// ❌ Don't do this
std::cout << "Processing session" << std::endl;

// ✅ Do this instead
LOG_INFO("Processing session {}", session_id);
```

## Modern C++ Features

### Concepts

Use concepts for template constraints:

```cpp
template <typename T>
concept Printable = requires(const T& t) {
    { t.ToString() } -> std::convertible_to<std::string>;
};

template <Printable T>
void LogObject(const T& obj) {
    LOG_INFO("Object: {}", obj.ToString());
}
```

### std::format

Always use `std::format` for string formatting:

```cpp
std::string ToString() const {
    return std::format(
        "SamplingParams(temperature={}, top_p={}, top_k={}, "
        "ignore_eos={}, max_tokens={})",
        temperature, top_p, top_k, ignore_eos, max_tokens);
}
```

### [[nodiscard]] and const

Mark functions appropriately:

```cpp
class Session {
public:
    [[nodiscard]] std::string ToString() const;
    [[nodiscard]] std::size_t GetPromptLength() const;
    [[nodiscard]] bool IsFinished() const;
    
    // Non-const modifiers
    void SetStatus(SessionStatus status /*[in]*/);
    void AppendTokenId(TokenId token_id /*[in]*/);
};
```

### Parameter Direction Annotations

Document all parameters with direction annotations:

```cpp
void UpdateSessionState(
    const std::string& session_id /*[in]*/,
    SessionStatus new_status /*[in]*/,
    std::vector<TokenId>& output_tokens /*[out]*/,
    SessionMetadata& metadata /*[inout]*/) {
    
    // Implementation
}
```

### Range-based for loops

Prefer range-based loops with auto:

```cpp
// ✅ Good
for (const auto& session : sessions) {
    LOG_INFO("Processing session {}", session->session_id);
}

// ✅ Also good for modification
for (auto& session : mutable_sessions) {
    session->SetStatus(SessionStatus::Running);
}
```

### Smart Pointers

Use smart pointers with descriptive type aliases:

```cpp
using ModelPtr = std::shared_ptr<BaseModel>;
using LayerPtr = std::shared_ptr<BaseLayer>;
using SessionManagerPtr = std::unique_ptr<BaseSessionManager>;

class ModelRunner {
private:
    ModelPtr model_;
    std::vector<LayerPtr> layers_;
    SessionManagerPtr session_manager_;
    
public:
    ModelRunner(ModelPtr model /*[in]*/, 
               std::vector<LayerPtr> layers /*[in]*/)
        : model_(std::move(model)),
          layers_(std::move(layers)) {
        
        ASSERT_VALID_POINTER_ARGUMENT(model_);
        ASSERT_VALID_ARGUMENTS(!layers_.empty(), "No layers provided");
    }
};
```

## Memory Management

### RAII and Smart Pointers

- Use `std::shared_ptr` for shared ownership
- Use `std::unique_ptr` for exclusive ownership  
- Always validate pointer arguments in constructors
- Prefer move semantics for performance

```cpp
class KVCacheStore {
private:
    std::size_t head_size_;
    std::size_t num_layers_;
    std::size_t num_kv_heads_;
    
public:
    KVCacheStore(std::size_t head_size,
                std::size_t num_layers,
                std::size_t num_kv_heads)
        : head_size_(head_size),
          num_layers_(num_layers),
          num_kv_heads_(num_kv_heads) {
        
        ASSERT_VALID_ARGUMENTS(head_size_ > 0, "Head size must be positive");
        ASSERT_VALID_ARGUMENTS(num_layers_ > 0, "Number of layers must be positive");
        ASSERT_VALID_ARGUMENTS(num_kv_heads_ > 0, "Number of KV heads must be positive");
    }
};
```

### Resource Management

Use RAII for all resources:

```cpp
class GPUMemoryPool {
private:
    void* gpu_memory_;
    std::size_t size_;
    
public:
    GPUMemoryPool(std::size_t size) : size_(size) {
        gpu_memory_ = allocateGPUMemory(size_);
        ASSERT_VALID_RUNTIME(gpu_memory_ != nullptr, 
            "Failed to allocate {}MB of GPU memory", size_ / 1024 / 1024);
    }
    
    ~GPUMemoryPool() {
        if (gpu_memory_) {
            freeGPUMemory(gpu_memory_);
        }
    }
    
    // Delete copy, allow move
    GPUMemoryPool(const GPUMemoryPool&) = delete;
    GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;
    GPUMemoryPool(GPUMemoryPool&& other) noexcept 
        : gpu_memory_(other.gpu_memory_), size_(other.size_) {
        other.gpu_memory_ = nullptr;
        other.size_ = 0;
    }
};
```

## Threading and Concurrency

### Thread Safety

Document thread safety requirements:

```cpp
class SessionManager {
private:
    mutable std::recursive_mutex mutex_;  // Allows recursive locking
    std::unordered_map<std::string, SessionPtr> session_map_;
    
public:
    // Thread-safe: Uses internal locking
    void AddSession(SessionPtr session /*[in]*/) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        ASSERT_VALID_POINTER_ARGUMENT(session);
        session_map_[session->session_id] = std::move(session);
    }
    
    // Thread-safe: Read-only access with lock
    [[nodiscard]] SessionPtr GetSession(const std::string& session_id /*[in]*/) const {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        auto it = session_map_.find(session_id);
        return (it != session_map_.end()) ? it->second : nullptr;
    }
};
```

### Atomic Operations

Use atomics for simple shared state:

```cpp
class MetricsCollector {
private:
    std::atomic<std::uint64_t> session_count_{0};
    std::atomic<std::uint64_t> tokens_processed_{0};
    
public:
    void IncrementSessions() noexcept {
        session_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    [[nodiscard]] std::uint64_t GetSessionCount() const noexcept {
        return session_count_.load(std::memory_order_relaxed);
    }
};
```

## Performance Guidelines

### Avoid Unnecessary Copies

```cpp
// ✅ Pass large objects by const reference
void ProcessLargeObject(const LargeObject& obj /*[in]*/) {
    // Implementation
}

// ✅ Return by value for move-constructible types
[[nodiscard]] std::vector<TokenId> GenerateTokens() {
    std::vector<TokenId> tokens;
    // Fill tokens...
    return tokens;  // RVO/move
}

// ✅ Use string_view for read-only string parameters
void LogMessage(std::string_view message /*[in]*/) {
    LOG_INFO("{}", message);
}
```

### Prefer Stack Allocation

```cpp
// ✅ Stack allocation when size is known
std::array<float, 1024> buffer;

// ✅ Heap allocation only when necessary
auto large_buffer = std::make_unique<std::array<float, 1024*1024>>();
```

### Move Semantics

```cpp
class ResourceHolder {
private:
    std::vector<ExpensiveResource> resources_;
    
public:
    // Accept by value and move
    void SetResources(std::vector<ExpensiveResource> resources /*[in]*/) {
        resources_ = std::move(resources);
    }
    
    // Return by value for move
    [[nodiscard]] std::vector<ExpensiveResource> ReleaseResources() {
        return std::move(resources_);
    }
};
```

## Testing and Documentation

### Unit Test Naming

```cpp
class SessionTest : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;
};

TEST_F(SessionTest, Constructor_ValidParameters_CreatesSession) {
    // Test implementation
}

TEST_F(SessionTest, AppendToken_ValidToken_UpdatesLength) {
    // Test implementation  
}

TEST_F(SessionTest, SetStatus_InvalidTransition_ThrowsException) {
    // Test implementation
}
```

### Documentation Comments

Use clear, concise documentation:

```cpp
/// @brief Manages the lifecycle and state of inference sessions
/// 
/// The SessionManager is responsible for tracking session state transitions,
/// managing memory allocation for sessions, and coordinating between the
/// scheduler and execution engines. It is thread-safe and supports concurrent
/// access from multiple threads.
///
/// @note All public methods are thread-safe unless otherwise noted
class SessionManager : public NonCopyable {
public:
    /// @brief Adds a new session to be managed
    /// @param session The session to add (must not be null)
    /// @throws std::invalid_argument if session is null
    /// @throws std::runtime_error if session ID already exists
    void AddSession(SessionPtr session /*[in]*/);
    
    /// @brief Retrieves a session by ID
    /// @param session_id The unique session identifier
    /// @return Pointer to session, or nullptr if not found
    /// @note Thread-safe for concurrent access
    [[nodiscard]] SessionPtr GetSession(const std::string& session_id /*[in]*/) const;
};
```

## Testing Guidelines

### Test File Structure

All test files must follow this structure and be placed in the `csrc/test/` directory:

```cpp
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
#include <gtest/gtest.h>
//==============================================================================
#include "commons/StdCommon.h"
#include "native/core/tokenizer/Tokenizer.h"
//==============================================================================
using setu::commons::tokenizer::Tokenizer;
//==============================================================================
namespace setu::test::commons {
```

### Test Class Naming

Use descriptive test class names with the `Test` suffix:

```cpp
class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;
    
    // Test data and helper methods
    const std::string filepath_ = "testdata/native/core/tokenizer/TokenizerTest/tokenizer.json";
    TokenizerPtr tokenizer_;
};

class CacheManagerTest : public ::testing::Test {
    // Test implementation
};

class SessionSchedulerTest : public ::testing::Test {
    // Test implementation
};
```

### Test Method Naming

Use the pattern: `MethodName_Condition_ExpectedResult`

```cpp
TEST_F(TokenizerTest, BasicTokenizerTest) {
    // Tests basic encode/decode functionality
}

TEST_F(TokenizerTest, PartialDecode_EmptyTokenList_ReturnsEmptyResult) {
    // Tests edge case with empty input
}

TEST_F(TokenizerTest, PartialDecode_ValidTokens_ReturnsCorrectText) {
    // Tests normal operation with valid input
}

TEST_F(CacheManagerTest, AllocatePage_SufficientMemory_ReturnsValidPageId) {
    // Tests successful page allocation
}

TEST_F(CacheManagerTest, AllocatePage_InsufficientMemory_ThrowsException) {
    // Tests error handling for out-of-memory condition
}
```

### Test Data Organization

Use structured test data and avoid magic numbers:

```cpp
class TransferEngineUtilsTest : public ::testing::Test {
protected:
    // Organized test data structure
    struct TensorTestData {
        std::vector<std::int64_t> tensor_shape;
        std::size_t num_splits;
        std::vector<std::size_t> page_list;
        std::string description;
    };

    const std::vector<TensorTestData> test_cases_ = {
        {{4, 32, 128}, 2, {0, 1, 2, 3}, "Small tensor with 2 splits"},
        {{8, 64, 256}, 4, {0, 1, 2, 3, 4, 5, 6, 7}, "Medium tensor with 4 splits"},
        {{16, 128, 512}, 8, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 
         "Large tensor with 8 splits"}
    };
    
    // Helper constants
    static constexpr std::size_t kDefaultPageSize = 4096;
    static constexpr std::size_t kMaxTensorSize = 1024 * 1024;
};
```

### Type System Compliance in Tests

Always use fixed-width integer types and proper type aliases:

```cpp
TEST_F(TokenizerTest, BasicEncodeDecode) {
    // ✅ Good - using fixed-width types
    const std::int32_t expected_token_count = 5;
    const std::uint64_t max_session_length = 1024;
    const std::size_t buffer_size = 256;
    
    // ❌ Avoid - ambiguous types
    // int count = 5;
    // long length = 1024;
    // unsigned size = 256;
    
    // Test implementation using proper types
    std::vector<std::int32_t> token_ids = tokenizer_->Encode(input_text);
    ASSERT_EQ(token_ids.size(), expected_token_count);
}
```

### Comprehensive Test Coverage

Write tests for all major code paths:

```cpp
class TokenRangeTrackerTest : public ::testing::Test {
    // Test all functionality comprehensively
};

// Test normal operation
TEST_F(TokenRangeTrackerTest, BasicFunctionality) {
    // Tests initialization, getting unprocessed ranges, and updating ranges
    const std::size_t total_tokens = 10;
    const std::size_t first_batch_size = 5;
    
    TokenRangeTracker tracker(total_tokens, TokenRangeState::Unprocessed);
    
    // Test initial state
    TokenRange range = tracker.GetNextUnprocessedRange();
    EXPECT_EQ(range.start, 0);
    EXPECT_EQ(range.end, total_tokens);
    
    // Test after processing first batch
    tracker.UpdateRange(0, first_batch_size, TokenRangeState::Processed);
    range = tracker.GetNextUnprocessedRange();
    EXPECT_EQ(range.start, first_batch_size);
    EXPECT_EQ(range.end, total_tokens);
}

// Test edge cases
TEST_F(TokenRangeTrackerTest, EmptyTracker) {
    TokenRangeTracker tracker(0, TokenRangeState::Unprocessed);
    
    EXPECT_EQ(tracker.GetLength(), 0);
    auto range = tracker.GetNextUnprocessedRange();
    EXPECT_EQ(range.state, TokenRangeState::Unavailable);
}

// Test complex scenarios
TEST_F(TokenRangeTrackerTest, UpdateRangeOverlappingMultiple) {
    // Tests complex overlapping range updates
    TokenRangeTracker tracker(0, TokenRangeState::Unprocessed);
    tracker.AppendRange(5, TokenRangeState::Unprocessed);
    tracker.AppendRange(5, TokenRangeState::Processed);
    tracker.AppendRange(5, TokenRangeState::Unavailable);
    
    // Complex update spanning multiple ranges
    tracker.UpdateRange(3, 12, TokenRangeState::Processed);
    
    auto ranges = tracker.GetTokenRanges();
    EXPECT_EQ(ranges.size(), 3);
    // Verify the resulting range structure
}
```

### Test Documentation

Document test intent and expected behavior:

```cpp
/**
 * @brief Test basic tokenizer encode/decode functionality
 * 
 * This test verifies that text can be encoded to token IDs and decoded back
 * to the original text without loss of information.
 */
TEST_F(TokenizerTest, BasicTokenizerTest) {
    const std::string filepath = "testdata/native/core/tokenizer/TokenizerTest/tokenizer.json";
    TokenizerPtr tokenizer = Tokenizer::FromPath(filepath);
    ASSERT_TRUE(tokenizer != nullptr) 
        << "Tokenizer should be successfully created from path: " << filepath;

    const std::string input_text = "hello, there";
    std::vector<std::int32_t> token_ids = tokenizer->Encode(input_text);
    std::string decoded_text = tokenizer->Decode(token_ids);

    ASSERT_EQ(input_text, decoded_text)
        << "Decoded text should match original input text";
}

/**
 * @brief Test partial decoding functionality
 * 
 * This test verifies that the tokenizer can incrementally decode tokens
 * as they are added, handling unicode replacement characters correctly.
 * The count refers to the number of times an intermediate decoding
 * ended with the unicode replacement character.
 */
TEST_F(TokenizerTest, PartialDecodeTest) {
    // Implementation with clear intent documentation
}
```

### Test Error Messages

Provide clear, actionable error messages:

```cpp
TEST_F(CacheManagerTest, AllocatePhysicalPages) {
    const std::size_t num_pages = 10;
    const std::vector<std::int32_t> page_ids = manager_->AllocatePhysicalPages(num_pages);
    
    ASSERT_EQ(page_ids.size(), num_pages) 
        << "Expected " << num_pages << " pages, got " << page_ids.size();
    
    for (std::size_t i = 0; i < page_ids.size(); ++i) {
        EXPECT_GE(page_ids[i], 0) 
            << "Page ID at index " << i << " should be non-negative, got " << page_ids[i];
    }
}
```

### Test Data Files

Organize test data in the `csrc/test/testdata/` directory:

```
csrc/test/testdata/
├── native/
│   ├── TokenizerTest/
│   │   └── tokenizer.json
│   ├── ModelTest/
│   │   ├── config.json
│   │   └── weights.bin
│   └── SchedulerTest/
│       └── sessions.json
└── kernels/
    ├── input_data.bin
    └── expected_output.bin
```

### Performance Tests

Include performance validation where appropriate:

```cpp
TEST_F(SchedulerTest, BatchFormation_LargeSessionCount_CompletesWithinTimeout) {
    const std::size_t num_sessions = 1000;
    const auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create and schedule many sessions
    Sessions sessions;
    for (std::size_t i = 0; i < num_sessions; ++i) {
        sessions.push_back(CreateTestSessionData(i));
    }
    
    scheduler_->ScheduleSessions(sessions);
    
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    EXPECT_LT(duration.count(), 100) 
        << "Scheduling " << num_sessions << " sessions took " << duration.count() 
        << "ms, expected < 100ms";
}
```

### Mock Objects and Test Doubles

Use Google Mock for testing complex interactions:

```cpp
class MockSessionManager : public BaseSessionManager {
public:
    MOCK_METHOD(void, AddSession, (SessionPtr session), (override));
    MOCK_METHOD(SessionPtr, GetSession, (const std::string& session_id), (const, override));
    MOCK_METHOD(void, RemoveSession, (const std::string& session_id), (override));
};

TEST_F(ControllerTest, ProcessSession_ValidSession_CallsSessionManager) {
    auto mock_manager = std::make_shared<MockSessionManager>();
    Controller controller(mock_manager);
    
    EXPECT_CALL(*mock_manager, AddSession(testing::_))
        .Times(1)
        .WillOnce(testing::Return());
    
    const auto session = CreateTestSessionData();
    controller.ProcessSession(session);
}
```

## Common Patterns

### Factory Functions

```cpp
class ModelFactory {
public:
    [[nodiscard]] static std::unique_ptr<BaseModel> CreateModel(
        const ModelConfig& config /*[in]*/) {
        
        ASSERT_VALID_ARGUMENTS(!config.model_name.empty(), 
            "Model name cannot be empty");
        
        // All models now use BaseModel with Python delegation
        return std::make_unique<BaseModel>(python_model, lm_head_weight);
        
        RAISE_RUNTIME_ERROR("Unsupported model type: {}", 
            static_cast<int>(config.model_type));
    }
};
```
