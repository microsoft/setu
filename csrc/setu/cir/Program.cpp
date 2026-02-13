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
#include "cir/Program.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::cir {
//==============================================================================

// ========================= Program Builder =================================

Value Program::EmitView(const Device& device, const setu::ir::ShardRef& handle,
                        Slice slice, torch::Dtype dtype) {
  ASSERT_VALID_ARGUMENTS(slice.size > 0, "View slice size must be positive");

  auto op_index = static_cast<std::uint32_t>(ops_.size());
  auto out = AllocateValue(device, slice.size, dtype, op_index);

  ops_.emplace_back(ViewOp{.out = out,
                           .device = device,
                           .handle = handle,
                           .slice = slice,
                           .dtype = dtype});

  return out;
}

Value Program::EmitAllocTmp(const Device& device, std::size_t size_elements,
                            torch::Dtype dtype) {
  ASSERT_VALID_ARGUMENTS(size_elements > 0, "alloc_tmp size must be positive");

  auto op_index = static_cast<std::uint32_t>(ops_.size());
  auto out = AllocateValue(device, size_elements, dtype, op_index);

  ops_.emplace_back(AllocTmpOp{.out = out,
                               .device = device,
                               .size_elements = size_elements,
                               .dtype = dtype});

  return out;
}

Value Program::EmitCopy(Value src, Value dst_in) {
  const auto& src_info = GetValueInfo(src);
  const auto& dst_info = GetValueInfo(dst_in);
  ASSERT_VALID_ARGUMENTS(src_info.size_elements == dst_info.size_elements,
                         "copy src size ({}) != dst size ({})",
                         src_info.size_elements, dst_info.size_elements);

  auto op_index = static_cast<std::uint32_t>(ops_.size());
  auto dst_out = AllocateValue(dst_info.device, dst_info.size_elements,
                               dst_info.dtype, op_index);

  ops_.emplace_back(CopyOp{.dst_out = dst_out, .src = src, .dst_in = dst_in});

  return dst_out;
}

Value Program::EmitPack(std::vector<Value> srcs, Value dst_in) {
  ASSERT_VALID_ARGUMENTS(!srcs.empty(), "pack requires at least one source");
  const auto& dst_info = GetValueInfo(dst_in);

  std::size_t total_src_size = 0;
  for (const auto& s : srcs) {
    total_src_size += GetValueInfo(s).size_elements;
  }
  ASSERT_VALID_ARGUMENTS(total_src_size == dst_info.size_elements,
                         "pack total src size ({}) != dst size ({})",
                         total_src_size, dst_info.size_elements);

  auto op_index = static_cast<std::uint32_t>(ops_.size());
  auto dst_out = AllocateValue(dst_info.device, dst_info.size_elements,
                               dst_info.dtype, op_index);

  ops_.emplace_back(
      PackOp{.dst_out = dst_out, .srcs = std::move(srcs), .dst_in = dst_in});

  return dst_out;
}

std::vector<Value> Program::EmitUnpack(Value src, std::vector<Value> dst_ins) {
  ASSERT_VALID_ARGUMENTS(!dst_ins.empty(),
                         "unpack requires at least one destination");
  const auto& src_info = GetValueInfo(src);

  std::size_t total_dst_size = 0;
  for (const auto& d : dst_ins) {
    total_dst_size += GetValueInfo(d).size_elements;
  }
  ASSERT_VALID_ARGUMENTS(src_info.size_elements == total_dst_size,
                         "unpack src size ({}) != total dst size ({})",
                         src_info.size_elements, total_dst_size);

  auto op_index = static_cast<std::uint32_t>(ops_.size());
  std::vector<Value> dst_outs;
  dst_outs.reserve(dst_ins.size());

  for (const auto& d : dst_ins) {
    const auto& d_info = GetValueInfo(d);
    dst_outs.push_back(AllocateValue(d_info.device, d_info.size_elements,
                                     d_info.dtype, op_index));
  }

  ops_.emplace_back(UnpackOp{
      .dst_outs = dst_outs, .src = src, .dst_ins = std::move(dst_ins)});

  return dst_outs;
}

// ========================= Query API =======================================

const ValueInfo& Program::GetValueInfo(Value v) const {
  ASSERT_VALID_ARGUMENTS(v.id < value_info_.size(),
                         "Value %{} out of range (num_values={})", v.id,
                         value_info_.size());
  return value_info_[v.id];
}

const Operation& Program::GetDefiningOp(Value v) const {
  const auto& info = GetValueInfo(v);
  ASSERT_VALID_ARGUMENTS(info.def_op_index < ops_.size(),
                         "def_op_index {} out of range (num_ops={})",
                         info.def_op_index, ops_.size());
  return ops_[info.def_op_index];
}

// ========================= Debug ===========================================

std::string Program::ToString() const {
  return std::format("Program(ops={}, values={})", ops_.size(),
                     value_info_.size());
}

std::string Program::Dump() const {
  std::string result;
  for (std::size_t i = 0; i < ops_.size(); ++i) {
    result += std::format("  [{}] {}\n", i, ops_[i].ToString());
  }
  return result;
}

// ========================= Private =========================================

Value Program::AllocateValue(const Device& device, std::size_t size_elements,
                             torch::Dtype dtype, std::uint32_t def_op_index) {
  Value v{next_value_id_++};
  value_info_.push_back(ValueInfo{.device = device,
                                  .size_elements = size_elements,
                                  .dtype = dtype,
                                  .def_op_index = def_op_index});
  return v;
}

// ========================= ProgramRewriter =================================

ProgramRewriter::ProgramRewriter(const Program& source) : source_(source) {}

Value ProgramRewriter::Lookup(Value old_value) const {
  auto it = value_map_.find(old_value);
  ASSERT_VALID_RUNTIME(it != value_map_.end(),
                       "Value {} not found in rewriter value map",
                       old_value.ToString());
  return it->second;
}

void ProgramRewriter::MapValue(Value old_value, Value new_value) {
  value_map_[old_value] = new_value;
}

void ProgramRewriter::CloneOp(std::size_t op_index) {
  ASSERT_VALID_ARGUMENTS(op_index < source_.NumOperations(),
                         "op_index {} out of range (num_ops={})", op_index,
                         source_.NumOperations());

  const auto& op = source_.Operations()[op_index];

  std::visit(
      [&](const auto& concrete_op) {
        using T = std::decay_t<decltype(concrete_op)>;

        if constexpr (std::is_same_v<T, ViewOp>) {
          auto new_val =
              target_.EmitView(concrete_op.device, concrete_op.handle,
                               concrete_op.slice, concrete_op.dtype);
          MapValue(concrete_op.out, new_val);

        } else if constexpr (std::is_same_v<T, AllocTmpOp>) {
          auto new_val = target_.EmitAllocTmp(
              concrete_op.device, concrete_op.size_elements, concrete_op.dtype);
          MapValue(concrete_op.out, new_val);

        } else if constexpr (std::is_same_v<T, CopyOp>) {
          auto new_val = target_.EmitCopy(Lookup(concrete_op.src),
                                          Lookup(concrete_op.dst_in));
          MapValue(concrete_op.dst_out, new_val);

        } else if constexpr (std::is_same_v<T, PackOp>) {
          std::vector<Value> new_srcs;
          new_srcs.reserve(concrete_op.srcs.size());
          for (const auto& s : concrete_op.srcs) {
            new_srcs.push_back(Lookup(s));
          }
          auto new_val =
              target_.EmitPack(std::move(new_srcs), Lookup(concrete_op.dst_in));
          MapValue(concrete_op.dst_out, new_val);

        } else if constexpr (std::is_same_v<T, UnpackOp>) {
          std::vector<Value> new_dst_ins;
          new_dst_ins.reserve(concrete_op.dst_ins.size());
          for (const auto& d : concrete_op.dst_ins) {
            new_dst_ins.push_back(Lookup(d));
          }
          auto new_vals = target_.EmitUnpack(Lookup(concrete_op.src),
                                             std::move(new_dst_ins));
          for (std::size_t i = 0; i < concrete_op.dst_outs.size(); ++i) {
            MapValue(concrete_op.dst_outs[i], new_vals[i]);
          }
        }
      },
      op.op);
}

Program ProgramRewriter::Finish() { return std::move(target_); }

//==============================================================================
}  // namespace setu::cir
//==============================================================================
