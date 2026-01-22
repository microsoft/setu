#pragma once

#include "commons/StdCommon.h"
#include "commons/Types.h"
#include "commons/datatypes/CopySpec.h"
#include "commons/datatypes/TensorShardRef.h"
#include "commons/datatypes/TensorShardSpec.h"
#include "commons/enums/Enums.h"
#include "commons/utils/ZmqHelper.h"

namespace setu::client {
using setu::commons::ClientRank;
using setu::commons::CopyOperationId;
using setu::commons::datatypes::CopySpec;
using setu::commons::datatypes::TensorShardRef;
using setu::commons::datatypes::TensorShardSpec;
using setu::commons::enums::ErrorCode;
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
