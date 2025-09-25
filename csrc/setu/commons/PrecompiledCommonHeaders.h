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
// Precompiled Headers for Native Module
// Contains heavy headers commonly used across native C++ functionality
//==============================================================================
#pragma once

// Standard library common headers
#include "commons/StdCommon.h"

// Heavy PyTorch headers (major compilation cost)
#include "commons/TorchCommon.h"

// Boost headers (threading and UUID functionality)
#include "commons/BoostCommon.h"

// ZMQ headers (used in all ZMQ functionality)
#include "commons/ZmqCommon.h"

// CUDA headers (used in all CUDA functionality)
#include <cuda_runtime_api.h>  // NOLINT

// Common Setu headers (stable interfaces)
#include "commons/Logging.h"
//==============================================================================
