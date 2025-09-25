# Setu Project - Coding Agent Rules

You are a coding agent helping with the Setu project, a high-performance inference engine for large language models. This is a complex C++/Python hybrid project requiring strict adherence to coding standards.

## Project Context
Setu is a production-grade inference engine with:
- Core engine in modern C++20/23 with Python bindings via pybind11
- PyTorch integration for model execution
- Sophisticated scheduling and parallelism features
- Comprehensive metrics and monitoring

## Setup Options

### Option 1: VS Code Devcontainer (Recommended)
Setu supports development using VS Code devcontainers for a consistent, pre-configured environment:

1. Install [Docker](https://www.docker.com/products/docker-desktop) and [VS Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Clone the repository and open in VS Code
4. Reopen in container when prompted
5. Use VS Code's built-in build tasks (Terminal > Run Build Task...)

### Option 2: Manual Setup
```bash
# Clone repository
git clone https://github.com/project-vajra/setu
cd setu

# Install dependencies and build
make setup/dependencies
make build
```

### Option 3: Using Conda/Mamba
```bash
# Install mamba (if not available)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# Create and activate environment
make setup/environment
make setup/activate  # Shows activation command
make setup/dependencies
make build
```

## Build System (MANDATORY)
**ALWAYS use the Makefile - NEVER run cmake directly**

### Essential Build Commands
```bash
# Setup and dependencies
make setup/dependencies        # Install all dependencies
make setup/environment        # Create conda environment
make setup/activate           # Show activation command
make setup/check              # Check system dependencies

# Building
make build                    # Intelligent build (auto-detects what's needed)
make build/test              # Build tests intelligently

# Testing
make test                    # Run all tests
make test/unit              # Run unit tests
make test/integration       # Run integration tests
make test/ctest             # Run C++ tests with Google Test
make test/pytest            # Run Python tests with pytest

# Failed-only testing (for faster iteration)
make test/unit-failed-only     # Rerun failed unit tests (Python + C++)
make test/pyunit-failed-only   # Rerun failed Python unit tests only
make test/ctest-failed-only    # Rerun failed C++ tests only

# Code quality
make lint                   # Lint code
make format                 # Format Python code with Black

# Utilities
make clean                  # Clean build artifacts
make ccache/stats          # Show ccache statistics
make help                  # Show all available targets

# Debugging
make debug/enable-cores     # Enable core dumps (partial functionality)
make debug/enable-cores-sudo # Enable core dumps (recommended, full privileges)
make debug/crash           # Analyze crashes using gdb
make debug/deadlock        # Analyze deadlocks using py-spy and gdb

# Environment management
make setup/update-environment    # Update conda environment
```

### Build Workflow
1. **First-time setup**: `make setup/dependencies && make build`
2. **Daily development**: `make build` (auto-detects what's needed)
3. **After adding new files**: `make build` (full rebuild automatically)
4. **Quick iterations**: `make build/native_incremental` (fastest)
5. **Testing during development**: `make test/unit-failed-only` (rerun only failed tests)
6. **Before commits**: `make format && make lint && make test`

The build system intelligently detects what needs to be built and chooses the optimal strategy automatically. Use failed-only test targets during development for faster iteration cycles.

## C++ Coding Standards

### File Header (EXACT FORMAT REQUIRED)
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

// Implementation

//==============================================================================
}  // namespace setu::commons
//==============================================================================
```

### Common Header Files
Setu uses centralized header files for consistency and compilation efficiency:

- **StdCommon.h**: All standard C++ headers (memory, vector, algorithm, etc.)
- **TorchCommon.h**: PyTorch integration headers and CUDA support
- **BoostCommon.h**: Boost utilities including Queue<T>, PriorityQueue<T>

### Naming Conventions (ABSOLUTELY STRICT)
```cpp
// Classes and Types: PascalCase
class SessionManager;
struct TokenInfo;
enum class SessionStatus { kWaiting, kRunning, kFinished };

// Functions and Methods: PascalCase (NOT camelCase!)
void ProcessSession();
[[nodiscard]] std::string ToString() const;
void SetStatus(SessionStatus status /*[in]*/);
[[nodiscard]] bool IsFinished() const;

// Variables: snake_case
std::int32_t num_tokens;
std::string session_id;
TimeS arrival_time;

// Member Variables: snake_case with trailing underscore
private:
    SessionStatus state_;
    TokenIdsPtr prompt_token_ids_;
    bool is_finished_;

// Constants: kPascalCase
constexpr double kSamplingEps = 1e-5;
constexpr std::int32_t kMaxSessionLength = 4096;

// Namespaces: nested C++17 syntax matching directory structure
namespace setu::commons::scheduler { }
```

### Type System (CRITICAL REQUIREMENTS)
```cpp
// ALWAYS use fixed-width types
std::int32_t count;        // NOT int
std::uint64_t size;        // NOT unsigned long
std::size_t index;         // For container indices
std::uint32_t num_layers;  // NOT unsigned int

// NEVER use ambiguous types
// ❌ FORBIDDEN: int, long, unsigned, short

// Define domain-specific type aliases
using SessionId = std::string;
using TokenId = std::int32_t;
using BlockId = std::int32_t;
using PageId = std::int32_t;
using ReplicaId = std::int32_t;
using Rank = std::int32_t;
using LayerId = std::int32_t;
using TimeS = double;  // Time in seconds

// Container aliases
using TokenIds = std::vector<TokenId>;
using TokenIdsPtr = std::shared_ptr<TokenIds>;
using BlockTable = std::vector<BlockId>;
using PageTable = std::vector<PageId>;

// Smart pointer aliases
using SessionPtr = std::shared_ptr<const Session>;
using SessionPtr = std::shared_ptr<Session>;
using ModelPtr = std::shared_ptr<BaseModel>;

// AVOID std::pair and std::tuple - use named structs
// ❌ BAD
std::pair<std::int32_t, std::string> result;

// ✅ GOOD
struct TokenResult {
    TokenId id;
    std::string text;
};

struct SessionData {
    SessionId session_id;
    TokenIds tokens;
    SamplingParams params;
};
```

### Class Design Patterns
```cpp
// Use Setu class traits
#include "commons/ClassTraits.h"

class SessionManager : public NonCopyable {
    // Can be moved but not copied
};

class AbstractModelRunner : public NonCopyableNonMovable {
    // Cannot be copied or moved
};

class ModelUtils : public StaticClass {
    // Only static methods
public:
    static bool ValidateConfig(const ModelConfig& config);
};

// Constructor pattern with validation
Session::Session(
    const std::string session_id_param,
    const std::string prompt_param,
    const TokenIdsPtr prompt_token_ids_param,
    const std::size_t page_size_param,
    const TokenId eos_token_id_param,
    const TimeS arrival_time_param)
    : session_id(session_id_param),
      prompt(prompt_param),
      prompt_token_ids(prompt_token_ids_param),
      page_size(page_size_param),
      eos_token_id(eos_token_id_param),
      arrival_time(arrival_time_param),
      state_(SessionStatus::kWaiting),
      is_finished_(false) {
  
  ASSERT_VALID_POINTER_ARGUMENT(prompt_token_ids);
  ASSERT_VALID_ARGUMENTS(page_size > 0, "Page size must be positive");
  ASSERT_VALID_ARGUMENTS(!prompt.empty(), "Prompt cannot be empty");
}
```

### Error Handling and Validation
```cpp
// ALWAYS use Setu macros for validation
ASSERT_VALID_POINTER_ARGUMENT(ptr);
ASSERT_VALID_RUNTIME(condition, "Failed because: {}", reason);
ASSERT_VALID_ARGUMENTS(value > 0, "Value {} must be positive", value);

// For throwing errors
RAISE_RUNTIME_ERROR("Operation failed: {}", error_message);
RAISE_INVALID_ARGUMENTS_ERROR("Invalid parameter: {}", param_name);
THROW_NOT_IMPLEMENTED_ERROR("Feature {} not yet implemented", feature);

// NEVER use raw assertions or exceptions
// ❌ assert(ptr != nullptr);
// ❌ throw std::runtime_error("error");
```

### Logging (NO CONSOLE OUTPUT)
```cpp
// ALWAYS use Setu logging macros
LOG_DEBUG("Debug info: value = {}", value);
LOG_INFO("Processing {} sessions", num_sessions);
LOG_WARNING("Memory usage is high: {}MB", memory_mb);
LOG_ERROR("Failed to allocate memory for session {}", session_id);
LOG_CRITICAL("System is in unrecoverable state");

// NEVER use console output
// ❌ std::cout << "message" << std::endl;
// ❌ printf("message\n");
// ❌ std::cerr << "error";
```

### Modern C++ Features
```cpp
// Use concepts for template constraints
template <typename T>
concept Printable = requires(const T& t) {
    { t.ToString() } -> std::convertible_to<std::string>;
};

template <Printable T>
void LogObject(const T& obj) {
    LOG_INFO("Object: {}", obj.ToString());
}

// Use std::format for string formatting
std::string ToString() const {
    return std::format(
        "SamplingParams(temperature={:.2f}, top_p={:.2f}, "
        "top_k={}, max_tokens={})",
        temperature, top_p, top_k, max_tokens);
}

// Use [[nodiscard]] and const correctness
[[nodiscard]] std::string GetId() const;
[[nodiscard]] bool IsFinished() const;
[[nodiscard]] std::size_t GetLength() const;

// Parameter direction annotations
void UpdateSessionState(
    const std::string& session_id /*[in]*/,
    SessionStatus new_status /*[in]*/,
    std::vector<TokenId>& output_tokens /*[out]*/,
    SessionMetadata& metadata /*[inout]*/);
```

### Memory Management
```cpp
// RAII with smart pointers
std::unique_ptr<BlockSpaceManager> block_manager_;
std::shared_ptr<TokenizerWrapper> tokenizer_;

// Define type aliases for clarity
using ModelPtr = std::shared_ptr<BaseModel>;
using SessionManagerPtr = std::unique_ptr<BaseSessionManager>;

// Move semantics for performance
CacheEngine(std::unique_ptr<BlockSpaceManager> block_manager,
           std::shared_ptr<TokenizerWrapper> tokenizer)
    : block_manager_(std::move(block_manager)),
      tokenizer_(std::move(tokenizer)) {
    
    ASSERT_VALID_POINTER_ARGUMENT(block_manager_);
    ASSERT_VALID_POINTER_ARGUMENT(tokenizer_);
}
```

### Threading and Concurrency
```cpp
class SessionManager {
private:
    mutable std::recursive_mutex mutex_;
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

// Use atomics for simple shared state
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

## Python Coding Standards

### Import Organization (EXACT ORDER)
```python
# 1. Standard library imports (alphabetical)
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any

# 2. Third-party imports (alphabetical)
import numpy as np
import torch
from transformers import AutoTokenizer

# 3. Setu native imports (C++ bindings)
from setu._native.core.configs import ModelConfig as ModelConfig_C
from setu._native.engine import InferenceEngine as InferenceEngine_C
from setu._native.core.datatypes import Session, SamplingParams

# 4. Setu Python imports (alphabetical)
from setu.core.configs.base_poly_config import BasePolyConfig
from setu.core.configs.model_config import ModelConfig
from setu.core.configs.parallel_config import ParallelConfig
from setu.logger import init_logger
from setu.utils.dataclasses import frozen_dataclass
```

### Naming and Type Hints
```python
# Variables and functions: snake_case
model_config: ModelConfig = load_config()
session_length: int = 2048
is_ready: bool = False

def process_sessions(sessions: List[Session]) -> List[SamplerOutput]:
    """Process a batch of sessions."""
    pass

def schedule_sessions(
    sessions: List[Session],
    available_blocks: int,
    current_time: float,
    priority_weights: Optional[Dict[str, float]] = None
) -> Tuple[List[Session], List[Session]]:
    """Schedule sessions based on availability and priorities."""
    pass

# Classes: PascalCase
class InferenceEngine:
    """Main inference engine implementation."""
    pass

# Constants: UPPER_SNAKE_CASE
MAX_SEQUENCE_LENGTH: int = 4096
DEFAULT_TEMPERATURE: float = 1.0
SUPPORTED_MODELS: List[str] = ["llama", "mistral", "mixtral"]

# Type aliases for clarity
SessionId = str
TokenId = int
SessionMapping = Dict[SessionId, Session]
BatchTokens = List[List[TokenId]]
```

### Configuration Classes
```python
from setu.utils.dataclasses import frozen_dataclass
from dataclasses import field

@frozen_dataclass
class ModelConfig:
    """Configuration for model loading and execution."""
    
    model_name: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Name or path of the HuggingFace model."}
    )
    max_model_len: int = field(
        default=4096,
        metadata={"help": "Maximum session length."}
    )
    dtype: str = field(
        default="float16",
        metadata={"help": "Model data type (float16, bfloat16, float32)"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading model"}
    )
    
    def __post_init__(self) -> None:
        """Validate and create native handle."""
        self._validate_parameters()
        # Create C++ handle for interop
        object.__setattr__(self, 'native_handle', ModelConfig_C(
            model_name=self.model_name,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code
        ))
    
    def _validate_parameters(self) -> None:
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if self.dtype not in ["float16", "bfloat16", "float32"]:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
```

### Logging Pattern
```python
from setu.logger import init_logger

# Create module-level logger
logger = init_logger(__name__)

class InferenceEngine:
    """Main inference engine implementation."""
    
    def __init__(self, config: InferenceEngineConfig):
        logger.info("Initializing InferenceEngine with model: %s", 
                   config.model_config.model_name)
        self.config = config
        self._initialize_components()
        logger.info("InferenceEngine initialization complete")
    
    def process_session(self, session: Session) -> SessionOutput:
        """Process a single inference session."""
        logger.debug("Processing session %s with %d prompt tokens", 
                    session.session_id, len(session.prompt_token_ids))
        
        start_time = time.time()
        try:
            output = self._execute_inference(session)
            processing_time = time.time() - start_time
            logger.info("Session %s completed in %.3f seconds, generated %d tokens",
                       session.session_id, processing_time, len(output.output_token_ids))
            return output
        except Exception as e:
            logger.error("Session %s failed after %.3f seconds: %s",
                        session.session_id, time.time() - start_time, str(e))
            raise

# ABSOLUTELY FORBIDDEN
# ❌ print(f"Processing session {session_id}")
# ❌ print("Error:", str(error))
# ✅ logger.info("Processing session %s", session_id)
# ✅ logger.error("Error: %s", str(error))
```

### Design Patterns

#### Registry Pattern
```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, Dict, Type, Any

class SchedulerType(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
    FAIR_SHARE = "fair_share"

class BaseScheduler(ABC):
    """Base class for all schedulers."""
    
    @abstractmethod
    def schedule(self, sessions: List[Session]) -> List[Session]:
        """Schedule sessions for execution."""
        pass

class SchedulerRegistry:
    """Registry for scheduler implementations."""
    
    _registry: ClassVar[Dict[SchedulerType, Type[BaseScheduler]]] = {}
    
    @classmethod
    def register(cls, scheduler_type: SchedulerType, 
                implementation: Type[BaseScheduler]) -> None:
        """Register a scheduler implementation."""
        if scheduler_type in cls._registry:
            logger.warning("Overriding existing scheduler: %s", scheduler_type)
        cls._registry[scheduler_type] = implementation
    
    @classmethod
    def create(cls, scheduler_type: SchedulerType, config: Any) -> BaseScheduler:
        """Create scheduler instance."""
        if scheduler_type not in cls._registry:
            raise ValueError(f"Scheduler {scheduler_type} not registered")
        return cls._registry[scheduler_type](config)

# Usage
SchedulerRegistry.register(SchedulerType.FCFS, FCFSScheduler)
scheduler = SchedulerRegistry.create(SchedulerType.FCFS, config)
```

#### Native Handle Integration
```python
import setu._native as setu_native

class InferenceEngine:
    """Python wrapper for C++ InferenceEngine."""
    
    def __init__(self, config: InferenceEngineConfig):
        # Python-side initialization
        self.config = config
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_config.model_name,
            trust_remote_code=config.model_config.trust_remote_code
        )
        self._session_tracker: Dict[str, SessionInfo] = {}
        
        # Create C++ engine through native handle
        self._native_handle = setu_native.create_inference_engine(
            config.to_native_config()
        )
        logger.info("Created native inference engine")
    
    def add_session(self, prompt: str, 
                   sampling_params: Optional[SamplingParams] = None,
                   session_id: Optional[str] = None) -> str:
        """Add a new inference session."""
        # Generate ID if not provided
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Validate input
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Tokenize in Python
        prompt_token_ids = self._tokenizer.encode(prompt)
        
        # Use default sampling params if not provided
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # Create native session
        native_session = setu_native.Session(
            session_id=session_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params.to_native()
        )
        
        # Pass to C++ engine
        self._native_handle.add_session(native_session)
        
        # Track on Python side
        self._session_tracker[session_id] = SessionInfo(
            prompt=prompt,
            arrival_time=time.time(),
            status="pending"
        )
        
        logger.debug("Added session %s with %d tokens", 
                    session_id, len(prompt_token_ids))
        return session_id
    
    def __del__(self) -> None:
        """Ensure proper cleanup of native resources."""
        if hasattr(self, '_native_handle'):
            self._native_handle.stop()
            logger.debug("Cleaned up native inference engine")
```

## Testing Guidelines

### C++ Tests (Google Test)
```cpp
// Location: csrc/test/
// Naming: ClassNameTest.cpp

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
#include "native/core/session/Session.h"
//==============================================================================
using setu::commons::session::Session;
//==============================================================================

class SessionTest : public ::testing::Test {
protected:
    void SetUp() override {
        sample_tokens_ = std::make_shared<std::vector<std::int32_t>>(
            std::vector<std::int32_t>{1, 2, 3, 4, 5});
    }
    
    void TearDown() override {
        // Test cleanup
    }
    
    // Test data
    std::shared_ptr<std::vector<std::int32_t>> sample_tokens_;
    static constexpr std::size_t kDefaultPageSize = 16;
    static constexpr std::int32_t kEosTokenId = 2;
};

// Test naming: MethodName_Condition_ExpectedResult
TEST_F(SessionTest, Constructor_ValidParameters_CreatesSession) {
    const std::string session_id = "test_session";
    const std::string prompt = "test prompt";
    const TimeS arrival_time = 1234567890.0;
    
    auto session = std::make_shared<Session>(
        session_id, prompt, sample_tokens_, 
        kDefaultPageSize, kEosTokenId, arrival_time);
    
    ASSERT_EQ(session->session_id, session_id) 
        << "Session ID should match constructor parameter";
    ASSERT_EQ(session->GetPageSize(), kDefaultPageSize)
        << "Page size should match constructor parameter";
    ASSERT_FALSE(session->IsFinished())
        << "New session should not be finished";
}

TEST_F(SessionTest, AppendToken_ValidToken_UpdatesLength) {
    auto session = CreateTestSessionData();
    const std::int32_t new_token = 42;
    const std::size_t initial_length = session->GetLength();
    
    session->AppendToken(new_token);
    
    ASSERT_EQ(session->GetLength(), initial_length + 1)
        << "Length should increase by 1 after appending token";
    ASSERT_EQ(session->GetLastToken(), new_token)
        << "Last token should be the appended token";
}

TEST_F(SessionTest, Constructor_NullTokens_ThrowsException) {
    EXPECT_THROW({
        Session("test", "prompt", nullptr, kDefaultPageSize, 
                kEosTokenId, 0.0);
    }, std::invalid_argument) << "Constructor should throw on null tokens";
}

// Always use fixed-width types in tests
TEST_F(SessionTest, Performance_LargeSession_CompletesWithinTimeout) {
    const std::size_t large_size = 10000;
    const std::int32_t test_token = 123;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create large session and append many tokens
    auto session = CreateTestSessionData();
    for (std::size_t i = 0; i < large_size; ++i) {
        session->AppendToken(test_token);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_LT(duration.count(), 100) 
        << "Large session operations took " << duration.count() 
        << "ms, expected < 100ms";
}
```

### Python Tests (pytest)
```python
import pytest
import time
from unittest.mock import Mock, patch
from setu.core.configs.model_config import ModelConfig
from setu.engine.inference_engine import InferenceEngine

class TestModelConfig:
    """Test ModelConfig functionality."""
    
    def test_valid_config_creation(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_model_len=2048,
            dtype="float16"
        )
        
        assert config.model_name == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert config.max_model_len == 2048
        assert config.dtype == "float16"
    
    def test_invalid_max_model_len_raises_error(self):
        """Test that invalid max_model_len raises ValueError."""
        with pytest.raises(ValueError, match="max_model_len must be positive"):
            ModelConfig(
                model_name="test-model",
                max_model_len=-1
            )
    
    def test_empty_model_name_raises_error(self):
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            ModelConfig(
                model_name="   ",  # whitespace only
                max_model_len=2048
            )
    
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
    def test_supported_dtypes(self, dtype: str):
        """Test that supported dtypes work correctly."""
        config = ModelConfig(
            model_name="test-model",
            max_model_len=1024,
            dtype=dtype
        )
        assert config.dtype == dtype
    
    @pytest.fixture
    def sample_config(self) -> ModelConfig:
        """Provide sample config for tests."""
        return ModelConfig(
            model_name="test-model",
            max_model_len=1024,
            dtype="float16"
        )

class TestInferenceEngine:
    """Test InferenceEngine functionality."""
    
    @pytest.fixture
    def mock_config(self) -> ModelConfig:
        """Create a mock configuration for testing."""
        return ModelConfig(
            model_name="test-model",
            max_model_len=1024,
            dtype="float16"
        )
    
    @pytest.fixture
    def engine(self, mock_config: ModelConfig) -> InferenceEngine:
        """Create an InferenceEngine for testing."""
        with patch('setu._native.engine.create_inference_engine'):
            with patch('transformers.AutoTokenizer.from_pretrained'):
                return InferenceEngine(mock_config)
    
    def test_add_session_returns_valid_id(self, engine: InferenceEngine):
        """Test that add_session returns a valid session ID."""
        session_id = engine.add_session(
            prompt="Test prompt for inference",
            sampling_params=None
        )
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id.startswith("session_")
    
    def test_add_session_with_empty_prompt_raises_error(self, 
                                                      engine: InferenceEngine):
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            engine.add_session(prompt="")
    
    def test_add_session_tracks_session_info(self, engine: InferenceEngine):
        """Test that session info is tracked correctly."""
        prompt = "Test prompt"
        session_id = engine.add_session(prompt=prompt)
        
        assert session_id in engine._session_tracker
        assert engine._session_tracker[session_id].prompt == prompt
        assert engine._session_tracker[session_id].status == "pending"
```

## Coding Agent Specific Guidelines

### Code Generation Philosophy
- Generate production-ready code that follows all Setu standards
- Prefer complete implementations over partial code snippets
- Always include proper error handling and validation
- Focus on code correctness and performance considerations
- Suggest architectural improvements when appropriate

### Agent Interaction Patterns
- Provide step-by-step implementation guidance
- Break complex changes into manageable chunks
- Explain the reasoning behind design decisions
- Offer alternative approaches when multiple solutions exist
- Anticipate potential issues and provide preventive measures
- Recommend efficient testing workflows using failed-only targets for faster iteration

### Quality Assurance
- Validate all generated code against Setu naming conventions
- Ensure fixed-width types are used throughout C++ code
- Verify proper logging and error handling patterns
- Check for thread safety considerations in concurrent code
- Confirm compatibility with existing codebase patterns

### Debugging and Optimization
- Suggest performance optimizations aligned with Setu goals
- Recommend appropriate debugging strategies for complex issues
- Provide memory management best practices
- Identify potential bottlenecks in proposed solutions
- Suggest profiling techniques for performance analysis

## Common Pitfalls to Avoid

1. **Naming Convention Violations**
   - ❌ `getStatus()` → ✅ `GetStatus()`
   - ❌ `MaxTokens` → ✅ `max_tokens`
   - ❌ `K_SAMPLING_EPS` → ✅ `kSamplingEps`

2. **Type System Violations**
   - ❌ `int count` → ✅ `std::int32_t count`
   - ❌ `unsigned size` → ✅ `std::size_t size`
   - ❌ `std::pair<int, string>` → ✅ Named struct

3. **Output Violations**
   - ❌ `std::cout <<` → ✅ `LOG_INFO()`
   - ❌ `print()` → ✅ `logger.info()`

4. **Build System Violations**
   - ❌ `cmake ..` → ✅ `make build`
   - ❌ `g++ -c file.cpp` → ✅ `make build/native`

5. **Include Path Violations**
   - ❌ `#include "setu/commons/StdCommon.h"` → ✅ `#include "commons/StdCommon.h"`

## Code Review Checklist

Before submitting:
- [ ] Setup: Used `make setup/dependencies` if needed
- [ ] Build: Used `make build` successfully  
- [ ] Format: Ran `make format` for Python code
- [ ] Lint: Ran `make lint` for code quality
- [ ] Tests: All tests pass with `make test` (use `make test/unit-failed-only` during development)
- [ ] C++ naming: All functions use PascalCase (e.g., `GetStatus()`)
- [ ] C++ types: All integers use fixed-width types
- [ ] Structures: No std::pair/tuple - using named structs
- [ ] Logging: Using LOG_* macros (no cout/print)
- [ ] Python types: All functions have complete type hints
- [ ] Imports: Follow exact order specification
- [ ] Validation: Using ASSERT_VALID_* macros
- [ ] Documentation: Added/updated docstrings where needed
- [ ] Thread safety: Documented where applicable
- [ ] Native handles: Properly integrated for Python/C++ interop

## Agent Best Practices

### Code Generation
- Always generate complete, compilable code
- Include all necessary includes and imports
- Follow RAII principles in C++ code
- Use appropriate design patterns from Setu codebase
- DO NOT use try-catch or other error catching patterns unless absolutely necessory. When anything unexpected happens, the application should crash and we should know. Do not ignore errors. Add many assertions to surface any pre/post condition violations easily.
- DO NOT keep retain any legacy code paths while making refactoring changes.
- NO NOT hardcode dummy values, when unsure ask!

### Problem Solving
- Analyze requirements thoroughly before proposing solutions
- Consider performance implications of design choices
- Evaluate memory usage and optimization opportunities
- Think about extensibility and maintainability
- Assess integration points with existing components

### Communication
- Explain complex design decisions clearly
- Provide rationale for architectural choices
- Highlight potential risks and mitigation strategies
- Suggest testing strategies for new implementations
- Document any assumptions made during development

## Quick Reference

### Setup Commands
- `make setup/dependencies` - Install all dependencies
- `make setup/environment` - Create conda environment
- `make setup/activate` - Show activation command
- `make setup/check` - Check system dependencies

### Build Commands
- `make build` - Intelligent build (auto-detects what's needed)
- `make build/test` - Build tests intelligently
- `make clean` - Clean build artifacts

### Test Commands
- `make test` - Run all tests
- `make test/unit` - Run unit tests
- `make test/integration` - Run integration tests
- `make test/ctest` - Run C++ tests
- `make test/pytest` - Run Python tests

# Failed-only testing (for faster development iteration)
- `make test/unit-failed-only` - Rerun failed unit tests (Python + C++)
- `make test/pyunit-failed-only` - Rerun failed Python unit tests only
- `make test/ctest-failed-only` - Rerun failed C++ tests only

### Code Quality Commands
- `make lint` - Lint code
- `make format` - Format Python code

### Debug Commands  
- `make debug/enable-cores-sudo` - Enable core dumps (recommended)
- `make debug/crash` - Analyze crashes
- `make debug/deadlock` - Analyze deadlocks

### Key Patterns
- C++ functions: `PascalCase` (e.g., `ProcessSession()`)
- Python functions: `snake_case` (e.g., `process_session()`)
- Fixed-width types: `std::int32_t`, `std::uint64_t` (never `int`, `long`)
- Logging: `LOG_INFO()` (C++), `logger.info()` (Python)
- Validation: `ASSERT_VALID_POINTER_ARGUMENT(ptr)`
- Config classes: `@frozen_dataclass` with `__post_init__`

Remember: As a coding agent, prioritize code quality, correctness, and adherence to Setu standards. When uncertain about implementation details, reference existing Setu patterns and maintain consistency with the established codebase.

## Recent Architecture Changes (July 2025)

### Module Naming: Plural to Singular
All module and folder names have been updated from plural to singular form for consistency:
- `controllers` → `controller`
- `schedulers` → `scheduler` (now `batcher`)
- `routers` → `router`
- `prioritizers` → `prioritizer`
- `batchers` → `batcher`
- `trackers` → `tracker`

### Component Renaming for Clarity
Major components have been renamed to better reflect their functionality:
- **ReplicaScheduler** → **SessionBatcher**: Batches sessions within a single replica
- **ReplicasetScheduler** → **SessionRouter**: Routes sessions across multiple replicas
- **scheduler_config** → **batcher_config**: Configuration naming aligned with component

### Module Reorganization
- **SessionPrioritizer** moved from `controller/replicaset/session_prioritizer` to `controller/helpers/session_prioritizer`
  - Reflects its role as a shared helper component
  - Used by both replica and replicaset controllers

### Directory Structure
```
setu/
├── core/
│   ├── controller/
│   │   ├── helpers/
│   │   │   └── session_prioritizer/    # Shared prioritization logic
│   │   ├── replica/
│   │   │   └── session_batcher/        # Intra-replica batching
│   │   └── replicaset/
│   │       └── session_router/         # Inter-replica routing
```

### Import Path Updates
- C++: `#include "native/core/controller/helpers/session_prioritizer/AbstractSessionPrioritizer.h"`
- Python: `from setu.core.controller.helpers.session_prioritizer import AbstractSessionPrioritizer`

### Namespace Updates
All C++ namespaces updated to singular form:
- `namespace setu::commons::llm::controller::replica::session_batcher`
- `namespace setu::commons::llm::controller::replicaset::session_router`
- `namespace setu::commons::llm::controller::helpers::session_prioritizer` 