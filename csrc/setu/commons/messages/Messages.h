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
#include "commons/messages/GetTensorHandleRequest.h"
#include "commons/messages/GetTensorHandleResponse.h"
#include "commons/messages/RegisterTensorShardRequest.h"
#include "commons/messages/RegisterTensorShardResponse.h"
#include "commons/messages/SubmitCopyRequest.h"
#include "commons/messages/SubmitCopyResponse.h"
#include "commons/messages/WaitForCopyRequest.h"
#include "commons/messages/WaitForCopyResponse.h"
//==============================================================================
namespace setu::commons::messages {
//==============================================================================
/// @brief Requests from Client to NodeAgent.
using ClientRequest =
    std::variant<RegisterTensorShardRequest, SubmitCopyRequest,
                 WaitForCopyRequest, GetTensorHandleRequest>;

/// @brief Requests from NodeAgent to Coordinator.
using NodeAgentRequest = std::variant<RegisterTensorShardRequest,
                                      SubmitCopyRequest, WaitForCopyRequest>;

/// @brief All messages from Coordinator to NodeAgent (flattened).
/// This unified type enables a single, flat dispatch loop instead of
/// nested request/response handling.
using CoordinatorMessage =
    std::variant<AllocateTensorRequest, CopyOperationFinishedRequest,
                 ExecuteRequest, RegisterTensorShardResponse,
                 SubmitCopyResponse, WaitForCopyResponse>;

using Request = std::variant<RegisterTensorShardRequest, SubmitCopyRequest,
                             WaitForCopyRequest, GetTensorHandleRequest,
                             AllocateTensorRequest,
                             CopyOperationFinishedRequest, ExecuteRequest>;

using Response = std::variant<RegisterTensorShardResponse, SubmitCopyResponse,
                              WaitForCopyResponse, GetTensorHandleResponse,
                              AllocateTensorResponse,
                              CopyOperationFinishedResponse, ExecuteResponse>;
//==============================================================================
}  // namespace setu::commons::messages
//==============================================================================
