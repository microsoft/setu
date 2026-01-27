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
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/enums/Enums.h"
#include "commons/utils/TorchTensorIPC.h"
#include "commons/utils/ZmqHelper.h"

namespace setu::client {
using setu::commons::ClientRank;
using setu::commons::CopyOperationId;
using setu::commons::TensorName;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::enums::ErrorCode;
using setu::commons::utils::TensorIPCSpec;
using setu::commons::utils::ZmqContextPtr;
using setu::commons::utils::ZmqSocketPtr;

class Client {
 public:
  Client(ClientRank client_rank);
  ~Client();

  void Connect(const std::string& endpoint);

  void Disconnect();

  bool IsConnected() const;

  const std::string& GetEndpoint() const;

  std::optional<TensorShardRef> RegisterTensorShard(
      const TensorShardSpec& shard_spec);

  std::optional<CopyOperationId> SubmitCopy(const CopySpec& copy_spec);

  void WaitForCopy(CopyOperationId copy_op_id);

  TensorIPCSpec GetTensorHandle(TensorName tensor_name);

 private:
  ClientRank client_rank_;
  // Zmq context and sockets
  ZmqContextPtr zmq_context_;
  ZmqSocketPtr request_socket_;

  std::string endpoint_;
  bool is_connected_{false};
};
//==============================================================================
}  // namespace setu::client
