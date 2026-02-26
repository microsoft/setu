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
#include "messaging/AllocateTensorRequest.h"
#include "messaging/AllocateTensorResponse.h"
#include "messaging/BaseResponse.h"
#include "messaging/CopyOperationFinishedRequest.h"
#include "messaging/CopyOperationFinishedResponse.h"
#include "messaging/DeregisterShardsRequest.h"
#include "messaging/DeregisterShardsResponse.h"
#include "messaging/ExecuteProgramRequest.h"
#include "messaging/ExecuteProgramResponse.h"
#include "messaging/ExecuteRequest.h"
#include "messaging/ExecuteResponse.h"
#include "messaging/GetTensorHandleRequest.h"
#include "messaging/GetTensorHandleResponse.h"
#include "messaging/GetTensorSelectionRequest.h"
#include "messaging/GetTensorSelectionResponse.h"
#include "messaging/GetTensorSpecRequest.h"
#include "messaging/GetTensorSpecResponse.h"
#include "messaging/RegisterTensorShardCoordinatorResponse.h"
#include "messaging/RegisterTensorShardNodeAgentResponse.h"
#include "messaging/RegisterTensorShardRequest.h"
#include "messaging/SubmitCopyRequest.h"
#include "messaging/SubmitCopyResponse.h"
#include "messaging/SubmitPullRequest.h"
#include "messaging/WaitForCopyRequest.h"
#include "messaging/WaitForCopyResponse.h"
#include "messaging/WaitForShardAllocationRequest.h"
#include "messaging/WaitForShardAllocationResponse.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
/// @brief Requests from Client to NodeAgent.
using ClientRequest =
    std::variant<RegisterTensorShardRequest, SubmitCopyRequest,
                 SubmitPullRequest, WaitForCopyRequest, GetTensorHandleRequest,
                 WaitForShardAllocationRequest, GetTensorSelectionRequest,
                 DeregisterShardsRequest>;

/// @brief Requests from NodeAgent to Coordinator.
using NodeAgentRequest =
    std::variant<RegisterTensorShardRequest, SubmitCopyRequest,
                 SubmitPullRequest, ExecuteResponse, GetTensorSpecRequest,
                 DeregisterShardsRequest>;

/// @brief All messages from Coordinator to NodeAgent
using CoordinatorMessage =
    std::variant<AllocateTensorRequest, CopyOperationFinishedRequest,
                 ExecuteRequest, RegisterTensorShardCoordinatorResponse,
                 SubmitCopyResponse, WaitForCopyResponse, GetTensorSpecResponse,
                 DeregisterShardsResponse>;

using Request =
    std::variant<RegisterTensorShardRequest, SubmitCopyRequest,
                 SubmitPullRequest, WaitForCopyRequest, GetTensorHandleRequest,
                 AllocateTensorRequest, CopyOperationFinishedRequest,
                 ExecuteRequest, WaitForShardAllocationRequest>;

using Response =
    std::variant<RegisterTensorShardNodeAgentResponse, SubmitCopyResponse,
                 WaitForCopyResponse, GetTensorHandleResponse,
                 AllocateTensorResponse, CopyOperationFinishedResponse,
                 ExecuteResponse, WaitForShardAllocationResponse,
                 GetTensorSelectionResponse>;
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
