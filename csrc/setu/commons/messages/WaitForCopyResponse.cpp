#include "commons/messages/WaitForCopyResponse.h"

namespace setu::commons::messages {
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;

void WaitForCopyResponse::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  writer.WriteFields(request_id, error_code);
}

WaitForCopyResponse WaitForCopyResponse::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [request_id_val, error_code_val] =
      reader.ReadFields<RequestId, ErrorCode>();
  return WaitForCopyResponse(request_id_val, error_code_val);
}
}  // namespace setu::commons::messages
