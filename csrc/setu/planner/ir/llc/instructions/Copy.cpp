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
#include "planner/ir/llc/instructions/Copy.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================

std::string Copy::ToString() const {
  return std::format(
      "Copy(src_ref={}, src_offset_bytes={}, dst_ref={}, "
      "dst_offset_bytes={}, count={}, dtype={}, src_ptr={}, dst_ptr={})",
      src_ref.ToString(), src_offset_bytes, dst_ref.ToString(),
      dst_offset_bytes, count, static_cast<int>(dtype), src_ptr, dst_ptr);
}

void Copy::Serialize(BinaryBuffer& buffer) const {
  BinaryWriter writer(buffer);
  const auto src_ptr_value = reinterpret_cast<std::uintptr_t>(src_ptr);
  const auto dst_ptr_value = reinterpret_cast<std::uintptr_t>(dst_ptr);
  writer.WriteFields(src_ref, src_offset_bytes, dst_ref, dst_offset_bytes,
                     count, dtype, src_ptr_value, dst_ptr_value);
}

Copy Copy::Deserialize(const BinaryRange& range) {
  BinaryReader reader(range);
  auto [src_ref, src_offset_bytes, dst_ref, dst_offset_bytes, count, dtype,
        src_ptr_val, dst_ptr_val] =
      reader.ReadFields<BufferRef, std::size_t, BufferRef, std::size_t,
                        std::size_t, torch::Dtype, std::uintptr_t,
                        std::uintptr_t>();

  auto src_ptr = reinterpret_cast<DevicePtr>(src_ptr_val);
  auto dst_ptr = reinterpret_cast<DevicePtr>(dst_ptr_val);
  return Copy(std::move(src_ref), src_offset_bytes, std::move(dst_ref),
              dst_offset_bytes, count, dtype, src_ptr, dst_ptr);
}

void Copy::Embellish(
    const std::function<DevicePtr(const BufferRef&)>& resolver) {
  src_ptr = resolver(src_ref);
  dst_ptr = resolver(dst_ref);
}

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
