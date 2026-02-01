#include "commons/messages/SubmitCopyResponse.h"

namespace setu::commons::messages {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;

void SubmitCopyResponse::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(request_id, error_code);
}

SubmitCopyResponse SubmitCopyResponse::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [request_id_val, copy_op_id, error_code_val] =
      reader.ReadFields<RequestId, CopyOperationId, ErrorCode>();
  return SubmitCopyResponse(request_id_val, copy_op_id, error_code_val);
}
}  // namespace setu::commons::messages
