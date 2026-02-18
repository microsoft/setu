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
#include "planner/passes/CopySpecToCIR.h"
//==============================================================================
#include "commons/Logging.h"
#include "planner/TensorShardRangeView.h"
#include "planner/ir/ref/ShardRef.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================

using setu::planner::ShardBufferRange;
using setu::planner::TensorShardRangeView;
using setu::planner::ir::cir::Device;
using setu::planner::ir::cir::Slice;
namespace ref = setu::planner::ir::ref;

//==============================================================================

/// Tracks consumption progress within a single shard buffer range.
struct ShardBufferState {
  explicit ShardBufferState(ShardBufferRange range_param)
      : buf(range_param), consumed(0) {}

  [[nodiscard]] bool IsConsumed() const { return consumed == buf.range.length; }

  [[nodiscard]] std::size_t Remaining() const {
    return buf.range.length - consumed;
  }

  void Consume(std::size_t sz) {
    ASSERT_VALID_ARGUMENTS(
        sz <= Remaining(),
        "Tried to consume more than what the buffer has left");
    consumed += sz;
  }

  /// Current offset in elements from the start of the shard buffer.
  [[nodiscard]] std::size_t CurrentOffsetElements() const {
    return buf.range.start + consumed;
  }

  ShardBufferRange buf;
  std::size_t consumed;
};

//==============================================================================

cir::Program CopySpecToCIR::Run(const CopySpec& copy_spec,
                                MetaStore& metastore) {
  auto src_own = metastore.GetTensorMetadata(copy_spec.src_name)
                     ->GetOwnershipMap(copy_spec.src_selection);
  auto dst_own = metastore.GetTensorMetadata(copy_spec.dst_name)
                     ->GetOwnershipMap(copy_spec.dst_selection);

  auto src_view = TensorShardRangeView(src_own);
  auto dst_view = TensorShardRangeView(dst_own);
  ASSERT_VALID_RUNTIME(!src_view.empty() && !dst_view.empty(),
                       "Source and destination views must not be empty");

  auto src_it = src_view.begin();
  auto dst_it = dst_view.begin();

  cir::Program program;

  auto src = ShardBufferState(*src_it);
  auto dst = ShardBufferState(*dst_it);

  auto advance_if_consumed = [](ShardBufferState& state, auto& it, auto end) {
    if (state.IsConsumed()) {
      ++it;
      if (it != end) {
        state = ShardBufferState(*it);
      }
    }
  };

  while (src_it != src_view.end() && dst_it != dst_view.end()) {
    auto to_copy = std::min(src.Remaining(), dst.Remaining());

    auto src_device =
        Device(src.buf.metadata->owner, src.buf.metadata->spec.device);
    auto dst_device =
        Device(dst.buf.metadata->owner, dst.buf.metadata->spec.device);

    auto src_shard_ref =
        ref::ShardRef(src.buf.metadata->id, src.buf.metadata->spec.name,
                      src.buf.metadata->owner);
    auto dst_shard_ref =
        ref::ShardRef(dst.buf.metadata->id, dst.buf.metadata->spec.name,
                      dst.buf.metadata->owner);

    auto dtype = src.buf.metadata->spec.dtype;

    auto src_val = program.EmitView(
        src_device, src_shard_ref,
        Slice{.offset = src.CurrentOffsetElements(), .size = to_copy}, dtype);

    auto dst_val = program.EmitView(
        dst_device, dst_shard_ref,
        Slice{.offset = dst.CurrentOffsetElements(), .size = to_copy}, dtype);

    (void)program.EmitCopy(src_val, dst_val);

    src.Consume(to_copy);
    dst.Consume(to_copy);
    advance_if_consumed(src, src_it, src_view.end());
    advance_if_consumed(dst, dst_it, dst_view.end());
  }

  ASSERT_VALID_RUNTIME(
      src_it == src_view.end() && dst_it == dst_view.end(),
      "Exited the loop before all src and dst buffers were consumed!");

  return program;
}

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
