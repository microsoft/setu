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
#include "commons/Types.h"
//==============================================================================
#include "commons/messages/AllocateTensorRequest.h"
#include "commons/messages/AllocateTensorResponse.h"
#include "commons/messages/BaseResponse.h"
#include "commons/messages/CopyOperationFinishedRequest.h"
#include "commons/messages/CopyOperationFinishedResponse.h"
#include "commons/messages/ExecuteProgramRequest.h"
#include "commons/messages/ExecuteProgramResponse.h"
#include "commons/messages/ExecuteRequest.h"
#include "commons/messages/ExecuteResponse.h"
#include "commons/messages/RegisterTensorShardRequest.h"
#include "commons/messages/RegisterTensorShardResponse.h"
#include "commons/messages/SubmitCopyRequest.h"
#include "commons/messages/SubmitCopyResponse.h"
#include "commons/messages/WaitForCopyRequest.h"
#include "commons/messages/WaitForCopyResponse.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
// Request variants by source
//==============================================================================
// Requests from clients to NodeAgent
using AnyClientRequest = std::variant<RegisterTensorShardRequest,
                                      SubmitCopyRequest, WaitForCopyRequest>;

// Requests from coordinator to NodeAgent
using AnyCoordinatorRequest =
    std::variant<AllocateTensorRequest, CopyOperationFinishedRequest,
                 ExecuteRequest>;

// All request types (for generic handling if needed)
using AnyRequest = std::variant<RegisterTensorShardRequest, SubmitCopyRequest,
                                WaitForCopyRequest, AllocateTensorRequest,
                                CopyOperationFinishedRequest, ExecuteRequest>;
//==============================================================================
// AnyResponse - Variant of all response types
//==============================================================================
using AnyResponse =
    std::variant<RegisterTensorShardResponse, SubmitCopyResponse,
                 WaitForCopyResponse, AllocateTensorResponse,
                 CopyOperationFinishedResponse, ExecuteResponse>;
//==============================================================================
// Helper for std::visit with lambdas
//==============================================================================
template <class... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
