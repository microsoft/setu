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
#include "planner/targets/DataDependence.h"
//==============================================================================
#include "commons/Logging.h"
#include "planner/ir/cir/Operation.h"
#include "planner/ir/ref/RegisterRef.h"
#include "planner/targets/IntervalMap.h"
//==============================================================================
namespace setu::planner::targets {
//==============================================================================
namespace cir = setu::planner::ir::cir;
namespace ref = setu::planner::ir::ref;

namespace {

/// Per-Value resolved location. Computed as CIR is walked; consumed
/// when a data-moving op needs to know the regions its operands touch.
struct ViewInfo {
  setu::planner::Participant participant;
  ref::BufferRef buffer_ref;
  std::size_t offset_bytes;
  std::size_t count;  ///< element count
  torch::Dtype dtype;
};

/// Convert a (participant-local) ViewInfo into a Region [start, end) in
/// bytes.
Region RegionOf(const ViewInfo& v) {
  return Region{
      .participant = v.participant,
      .buffer_ref = v.buffer_ref,
      .start_bytes = v.offset_bytes,
      .end_bytes = v.offset_bytes +
                   v.count * static_cast<std::size_t>(torch::elementSize(v.dtype)),
      .dtype = v.dtype,
  };
}

/// Key into the per-(participant, buffer) IntervalMap tables. Two
/// accesses produce an edge only when they share this key and their
/// byte ranges overlap.
struct AccessKey {
  setu::planner::Participant participant;
  ref::BufferRef buffer_ref;

  bool operator==(const AccessKey& other) const {
    return participant == other.participant && buffer_ref == other.buffer_ref;
  }
};

struct AccessKeyHash {
  std::size_t operator()(const AccessKey& k) const noexcept {
    std::size_t h1 = std::hash<setu::planner::Participant>{}(k.participant);
    std::size_t h2 = ref::hash_value(k.buffer_ref);
    return h1 ^ (h2 << 1);
  }
};

AccessKey KeyOf(const Region& r) {
  return AccessKey{.participant = r.participant, .buffer_ref = r.buffer_ref};
}

using WriterMap =
    std::unordered_map<AccessKey, IntervalMap<std::uint32_t>, AccessKeyHash>;
using ReaderMap =
    std::unordered_map<AccessKey, IntervalMap<std::uint32_t>, AccessKeyHash>;

}  // namespace

//==============================================================================

DataDependence BuildDataDependence(
    const cir::Program& program,
    const std::optional<cir::RegisterAllocation>& reg_alloc) {
  DataDependence dag;

  std::unordered_map<cir::Value, ViewInfo> view_map;
  WriterMap writers;
  ReaderMap readers;

  auto add_node = [&](std::uint32_t op_idx, std::vector<Region> reads,
                      std::vector<Region> writes) {
    std::set<setu::planner::Participant> parts;
    for (const auto& r : reads) parts.insert(r.participant);
    for (const auto& w : writes) parts.insert(w.participant);

    auto node_idx = static_cast<std::uint32_t>(dag.nodes.size());
    dag.nodes.push_back(DataDependenceNode{
        .op_idx = op_idx,
        .reads = reads,
        .writes = writes,
        .participants = std::move(parts),
    });
    dag.preds.emplace_back();
    dag.succs.emplace_back();
    auto& my_preds = dag.preds.back();

    // RAW: each read picks up all prior writers overlapping its range;
    // the read is recorded for later WAR detection by future writes.
    for (const auto& r : dag.nodes[node_idx].reads) {
      auto& w_map = writers[KeyOf(r)];
      for (const auto& e : w_map.Overlaps(r.start_bytes, r.end_bytes)) {
        my_preds.insert(e.value);
      }
      readers[KeyOf(r)].Insert(r.start_bytes, r.end_bytes, node_idx);
    }

    // WAW + WAR: each write picks up both prior writers and readers
    // overlapping its range; the write then supersedes both maps on
    // its bytes (new writer owns them; prior reads through those
    // bytes are WAR-resolved).
    for (const auto& w : dag.nodes[node_idx].writes) {
      auto key = KeyOf(w);
      auto& w_map = writers[key];
      for (const auto& e : w_map.Overlaps(w.start_bytes, w.end_bytes)) {
        my_preds.insert(e.value);
      }
      auto& r_map = readers[key];
      for (const auto& e : r_map.Overlaps(w.start_bytes, w.end_bytes)) {
        my_preds.insert(e.value);
      }
      w_map.SupersedeRange(w.start_bytes, w.end_bytes);
      w_map.Insert(w.start_bytes, w.end_bytes, node_idx);
      r_map.SupersedeRange(w.start_bytes, w.end_bytes);
    }

    // Mirror preds into succs to keep the transpose in sync with edges
    // added above.
    for (auto p : my_preds) dag.succs[p].insert(node_idx);
  };

  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;

          // ---------- reference builders: update view_map only ----------
          if constexpr (std::is_same_v<T, cir::ViewOp>) {
            auto element_size = torch::elementSize(concrete.dtype);
            view_map.try_emplace(
                concrete.out,
                ViewInfo{.participant = concrete.device,
                         .buffer_ref = ref::BufferRef(concrete.handle),
                         .offset_bytes = concrete.slice.offset * element_size,
                         .count = concrete.slice.size,
                         .dtype = concrete.dtype});

          } else if constexpr (std::is_same_v<T, cir::AllocTmpOp>) {
            ASSERT_VALID_RUNTIME(
                reg_alloc.has_value() &&
                    reg_alloc->allocation[concrete.out.id].has_value(),
                "AllocTmpOp {} has no register allocation",
                concrete.out.ToString());
            const auto& phys_reg =
                reg_alloc->allocation[concrete.out.id].value();
            view_map.try_emplace(
                concrete.out,
                ViewInfo{
                    .participant = concrete.device,
                    .buffer_ref = ref::BufferRef(ref::RegisterRef(
                        phys_reg.register_index, concrete.device)),
                    .offset_bytes = 0,
                    .count = concrete.size_elements,
                    .dtype = concrete.dtype,
                });

          } else if constexpr (std::is_same_v<T, cir::SliceOp>) {
            auto it = view_map.find(concrete.src);
            ASSERT_VALID_RUNTIME(it != view_map.end(),
                                 "SliceOp src {} not in view_map",
                                 concrete.src.ToString());
            const auto& src = it->second;
            auto element_size = torch::elementSize(src.dtype);
            view_map.try_emplace(
                concrete.out,
                ViewInfo{
                    .participant = src.participant,
                    .buffer_ref = src.buffer_ref,
                    .offset_bytes =
                        src.offset_bytes + concrete.slice.offset * element_size,
                    .count = concrete.slice.size,
                    .dtype = src.dtype,
                });

          } else if constexpr (std::is_same_v<T, cir::ConsumeOp>) {
            auto it = view_map.find(concrete.src);
            ASSERT_VALID_RUNTIME(it != view_map.end(),
                                 "ConsumeOp src {} not in view_map",
                                 concrete.src.ToString());
            view_map.try_emplace(concrete.out, it->second);

            // ---------- data movers: emit a DataDependenceNode ----------
          } else if constexpr (std::is_same_v<T, cir::CopyOp>) {
            auto src_it = view_map.find(concrete.src);
            auto dst_it = view_map.find(concrete.dst_in);
            ASSERT_VALID_RUNTIME(
                src_it != view_map.end() && dst_it != view_map.end(),
                "CopyOp operands {} and {} must be resolvable",
                concrete.src.ToString(), concrete.dst_in.ToString());
            add_node(op_idx, {RegionOf(src_it->second)},
                     {RegionOf(dst_it->second)});
            view_map.try_emplace(concrete.dst_out, dst_it->second);

          } else if constexpr (std::is_same_v<T, cir::PackOp>) {
            auto dst_it = view_map.find(concrete.dst_in);
            ASSERT_VALID_RUNTIME(dst_it != view_map.end(),
                                 "PackOp dst_in {} not in view_map",
                                 concrete.dst_in.ToString());
            const auto& dst = dst_it->second;

            std::vector<Region> reads;
            std::vector<Region> writes;
            reads.reserve(concrete.srcs.size());
            writes.reserve(concrete.srcs.size());
            std::size_t running = dst.offset_bytes;
            for (const auto& src_val : concrete.srcs) {
              auto src_it = view_map.find(src_val);
              ASSERT_VALID_RUNTIME(src_it != view_map.end(),
                                   "PackOp src {} not in view_map",
                                   src_val.ToString());
              const auto& src = src_it->second;
              reads.push_back(RegionOf(src));

              auto size_bytes = src.count * static_cast<std::size_t>(
                                                torch::elementSize(src.dtype));
              writes.push_back(Region{
                  .participant = dst.participant,
                  .buffer_ref = dst.buffer_ref,
                  .start_bytes = running,
                  .end_bytes = running + size_bytes,
                  .dtype = src.dtype,
              });
              running += size_bytes;
            }
            add_node(op_idx, std::move(reads), std::move(writes));
            view_map.try_emplace(concrete.dst_out, dst_it->second);

          } else if constexpr (std::is_same_v<T, cir::UnpackOp>) {
            auto src_it = view_map.find(concrete.src);
            ASSERT_VALID_RUNTIME(src_it != view_map.end(),
                                 "UnpackOp src {} not in view_map",
                                 concrete.src.ToString());
            const auto& src = src_it->second;
            ASSERT_VALID_RUNTIME(
                concrete.dst_ins.size() == concrete.dst_outs.size(),
                "UnpackOp dst_ins and dst_outs size mismatch");

            std::vector<Region> reads;
            std::vector<Region> writes;
            reads.reserve(concrete.dst_ins.size());
            writes.reserve(concrete.dst_ins.size());
            std::size_t running = src.offset_bytes;
            for (std::size_t i = 0; i < concrete.dst_ins.size(); ++i) {
              auto dst_it = view_map.find(concrete.dst_ins[i]);
              ASSERT_VALID_RUNTIME(dst_it != view_map.end(),
                                   "UnpackOp dst_in {} not in view_map",
                                   concrete.dst_ins[i].ToString());
              const auto& dst = dst_it->second;
              writes.push_back(RegionOf(dst));

              auto size_bytes = dst.count * static_cast<std::size_t>(
                                                torch::elementSize(dst.dtype));
              reads.push_back(Region{
                  .participant = src.participant,
                  .buffer_ref = src.buffer_ref,
                  .start_bytes = running,
                  .end_bytes = running + size_bytes,
                  .dtype = dst.dtype,
              });
              running += size_bytes;

              view_map.try_emplace(concrete.dst_outs[i], dst_it->second);
            }
            add_node(op_idx, std::move(reads), std::move(writes));

          } else if constexpr (std::is_same_v<T, cir::AllGatherOp>) {
            ASSERT_VALID_RUNTIME(
                concrete.srcs.size() == concrete.dst_ins.size() &&
                    concrete.srcs.size() == concrete.dst_outs.size(),
                "AllGatherOp: srcs/dst_ins/dst_outs size mismatch");

            std::vector<Region> reads;
            std::vector<Region> writes;
            reads.reserve(concrete.srcs.size());
            writes.reserve(concrete.dst_ins.size());
            for (std::size_t i = 0; i < concrete.srcs.size(); ++i) {
              auto src_it = view_map.find(concrete.srcs[i]);
              auto dst_it = view_map.find(concrete.dst_ins[i]);
              ASSERT_VALID_RUNTIME(src_it != view_map.end() &&
                                       dst_it != view_map.end(),
                                   "AllGatherOp operand missing in view_map");
              reads.push_back(RegionOf(src_it->second));
              writes.push_back(RegionOf(dst_it->second));
              view_map.try_emplace(concrete.dst_outs[i], dst_it->second);
            }
            add_node(op_idx, std::move(reads), std::move(writes));
          }
        },
        op.op);
  }

  return dag;
}

//==============================================================================
}  // namespace setu::planner::targets
//==============================================================================
