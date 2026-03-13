#include "planner/passes/RegisterTiling.h"

#include "commons/Logging.h"

namespace setu::planner::passes {

using ChunkMap = std::unordered_map<cir::Value, std::vector<cir::Value>>;

/// Walks two contiguous partitions of the same logical buffer and groups
/// pieces from `from_vals` by their overlap with each item in `by_vals`.
///
/// Returns one vector of Value pieces per item in `by_vals`.  Each piece is
/// either the full `from` value (when the overlap covers it entirely) or a
/// freshly emitted slice.
///
/// `from_size_fn(val)` / `by_size_fn(val)` return element counts.
/// `resolve(val, offset, size, total)` maps a from-value + sub-range to a
/// target-program Value (handling Lookup and slicing).
template <typename FromSizeFn, typename BySizeFn, typename ResolveFn>
static std::vector<std::vector<cir::Value>> GroupPiecesByOverlap(
    const std::vector<cir::Value>& from_vals, FromSizeFn from_size_fn,
    const std::vector<cir::Value>& by_vals, BySizeFn by_size_fn,
    ResolveFn resolve) {
  std::vector<std::vector<cir::Value>> groups(by_vals.size());
  std::size_t fi = 0, fo = 0;
  for (std::size_t bi = 0; bi < by_vals.size(); ++bi) {
    auto by_remaining = by_size_fn(by_vals[bi]);
    while (by_remaining > 0) {
      auto from_total = from_size_fn(from_vals[fi]);
      auto take = std::min(by_remaining, from_total - fo);
      groups[bi].push_back(resolve(from_vals[fi], fo, take, from_total));
      by_remaining -= take;
      fo += take;
      if (fo == from_total) {
        fi++;
        fo = 0;
      }
    }
  }
  return groups;
}

/// Number of elements that fit in one register-sized chunk for a given dtype.
inline std::size_t NumElementsInChunk(std::size_t chunk_size_bytes,
                                      torch::Dtype dtype) {
  auto element_size = torch::elementSize(dtype);
  ASSERT_VALID_RUNTIME(chunk_size_bytes >= element_size,
                       "Element size {} exceeds chunk size {}", element_size,
                       chunk_size_bytes);
  return chunk_size_bytes / element_size;
}

cir::Program RegisterTiling::Run(cir::Program program, const PassContext& ctx) {
  // Use register size from PassContext if available, otherwise fall back to
  // the constructor parameter.
  auto chunk_size_bytes = chunk_size_bytes_;
  if (!ctx.register_sets.empty()) {
    chunk_size_bytes = ctx.register_sets.begin()->second.SizeBytes(0);
  }

  bool has_large_tmp = false;
  for (const auto& op : program.Operations()) {
    if (op.Type() == cir::OpType::kAllocTmp) {
      const auto& alloc = std::get<cir::AllocTmpOp>(op.op);
      if (alloc.size_elements >
          NumElementsInChunk(chunk_size_bytes, alloc.dtype)) {
        has_large_tmp = true;
        break;
      }
    }
  }
  if (!has_large_tmp) {
    return program;
  }

  ChunkMap chunk_map;
  auto rw = cir::ProgramRewriter(program);

  for (std::size_t i = 0; i < program.NumOperations(); ++i) {
    const auto& op = program.Operations()[i];

    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;

          if constexpr (std::is_same_v<T, cir::AllocTmpOp>) {
            auto chunk_elems =
                NumElementsInChunk(chunk_size_bytes, concrete.dtype);

            if (concrete.size_elements <= chunk_elems) {
              rw.CloneOp(i);
              return;
            }

            auto num_chunks =
                (concrete.size_elements + chunk_elems - 1) / chunk_elems;
            auto device = program.GetValueInfo(concrete.out).device;

            std::vector<cir::Value> chunks;
            chunks.reserve(num_chunks);
            for (std::size_t c = 0; c < num_chunks; ++c) {
              auto this_chunk = std::min(
                  chunk_elems, concrete.size_elements - c * chunk_elems);
              chunks.push_back(
                  rw.Target().EmitAllocTmp(device, this_chunk, concrete.dtype));
            }
            chunk_map[concrete.out] = std::move(chunks);
            return;
          }

          if constexpr (std::is_same_v<T, cir::PackOp>) {
            if (!chunk_map.contains(concrete.dst_in)) {
              rw.CloneOp(i);
              return;
            }

            const auto& dst_chunks = chunk_map.at(concrete.dst_in);
            auto src_size = [&](cir::Value v) {
              return program.GetValueInfo(v).size_elements;
            };
            auto chunk_size = [&](cir::Value v) {
              return rw.Target().GetValueInfo(v).size_elements;
            };
            auto resolve_src = [&](cir::Value v, std::size_t off,
                                   std::size_t sz, std::size_t total) {
              auto mapped = rw.Lookup(v);
              if (off == 0 && sz == total) return mapped;
              return rw.Target().EmitSlice(mapped, cir::Slice{off, sz});
            };

            auto groups = GroupPiecesByOverlap(
                concrete.srcs, src_size, dst_chunks, chunk_size, resolve_src);

            std::vector<cir::Value> result_chunks;
            result_chunks.reserve(dst_chunks.size());
            for (std::size_t c = 0; c < dst_chunks.size(); ++c) {
              result_chunks.push_back(
                  rw.Target().EmitPack(std::move(groups[c]), dst_chunks[c]));
            }
            chunk_map[concrete.dst_out] = std::move(result_chunks);
            return;
          }

          if constexpr (std::is_same_v<T, cir::UnpackOp>) {
            if (!chunk_map.contains(concrete.src)) {
              rw.CloneOp(i);
              return;
            }

            const auto& src_chunks = chunk_map.at(concrete.src);
            auto chunk_size = [&](cir::Value v) {
              return rw.Target().GetValueInfo(v).size_elements;
            };
            auto dst_size = [&](cir::Value v) {
              return program.GetValueInfo(v).size_elements;
            };
            auto resolve_chunk = [&](cir::Value v, std::size_t off,
                                     std::size_t sz, std::size_t total) {
              if (off == 0 && sz == total) return v;
              return rw.Target().EmitSlice(v, cir::Slice{off, sz});
            };

            auto groups =
                GroupPiecesByOverlap(src_chunks, chunk_size, concrete.dst_ins,
                                     dst_size, resolve_chunk);

            for (std::size_t d = 0; d < concrete.dst_ins.size(); ++d) {
              auto mapped_dst = rw.Lookup(concrete.dst_ins[d]);
              cir::Value dst_out;
              if (groups[d].size() == 1) {
                dst_out = rw.Target().EmitCopy(groups[d][0], mapped_dst);
              } else {
                dst_out =
                    rw.Target().EmitPack(std::move(groups[d]), mapped_dst);
              }
              rw.MapValue(concrete.dst_outs[d], dst_out);
            }
            return;
          }

          if constexpr (std::is_same_v<T, cir::CopyOp>) {
            bool src_chunked = chunk_map.contains(concrete.src);
            bool dst_chunked = chunk_map.contains(concrete.dst_in);

            if (!src_chunked && !dst_chunked) {
              rw.CloneOp(i);
              return;
            }

            // Resolve the chunk count from whichever side is chunked.
            const auto& ref_chunks = src_chunked
                                         ? chunk_map.at(concrete.src)
                                         : chunk_map.at(concrete.dst_in);
            auto num_chunks = ref_chunks.size();

            // Returns the i-th chunk for a value: either a pre-existing
            // chunk from the map, or a freshly emitted slice.
            auto chunk_or_slice = [&](cir::Value old_val, bool is_chunked,
                                      std::size_t c, std::size_t offset,
                                      std::size_t chunk_elements) {
              if (is_chunked) {
                return chunk_map.at(old_val)[c];
              }
              return rw.Target().EmitSlice(rw.Lookup(old_val),
                                           cir::Slice{offset, chunk_elements});
            };

            std::vector<cir::Value> result_chunks;
            result_chunks.reserve(num_chunks);
            std::size_t offset = 0;

            for (std::size_t c = 0; c < num_chunks; ++c) {
              auto chunk_elements =
                  rw.Target().GetValueInfo(ref_chunks[c]).size_elements;
              auto src_val = chunk_or_slice(concrete.src, src_chunked, c,
                                            offset, chunk_elements);
              auto dst_val = chunk_or_slice(concrete.dst_in, dst_chunked, c,
                                            offset, chunk_elements);
              result_chunks.push_back(rw.Target().EmitCopy(src_val, dst_val));
              offset += chunk_elements;
            }

            if (dst_chunked) {
              chunk_map[concrete.dst_out] = std::move(result_chunks);
            } else {
              auto new_dst_out =
                  rw.Target().EmitConsume(rw.Lookup(concrete.dst_in));
              rw.MapValue(concrete.dst_out, new_dst_out);
            }
            return;
          }

          if constexpr (std::is_same_v<T, cir::SliceOp>) {
            if (chunk_map.contains(concrete.src)) {
              const auto& src_chunks = chunk_map.at(concrete.src);
              std::vector<cir::Value> result_chunks;

              std::size_t slice_start = concrete.slice.offset;
              std::size_t slice_end = slice_start + concrete.slice.size;
              std::size_t global_offset = 0;

              for (const auto& chunk_val : src_chunks) {
                auto chunk_elems =
                    rw.Target().GetValueInfo(chunk_val).size_elements;
                auto chunk_start = global_offset;
                auto chunk_end = global_offset + chunk_elems;

                auto overlap_start = std::max(slice_start, chunk_start);
                auto overlap_end = std::min(slice_end, chunk_end);

                if (overlap_start < overlap_end) {
                  auto local_offset = overlap_start - chunk_start;
                  auto overlap_size = overlap_end - overlap_start;

                  if (local_offset == 0 && overlap_size == chunk_elems) {
                    result_chunks.push_back(chunk_val);
                  } else {
                    result_chunks.push_back(rw.Target().EmitSlice(
                        chunk_val, cir::Slice{local_offset, overlap_size}));
                  }
                }
                global_offset += chunk_elems;
              }
              chunk_map[concrete.out] = std::move(result_chunks);
              return;
            }
          }

          if constexpr (std::is_same_v<T, cir::ConsumeOp>) {
            if (chunk_map.contains(concrete.src)) {
              const auto& src_chunks = chunk_map.at(concrete.src);
              std::vector<cir::Value> result_chunks;
              result_chunks.reserve(src_chunks.size());
              for (const auto& chunk_val : src_chunks) {
                result_chunks.push_back(rw.Target().EmitConsume(chunk_val));
              }
              chunk_map[concrete.out] = std::move(result_chunks);
              return;
            }
          }

          if constexpr (std::is_same_v<T, cir::AllGatherOp>) {
            for (const auto& s : concrete.srcs) {
              ASSERT_VALID_RUNTIME(
                  !chunk_map.contains(s),
                  "RegisterTiling: AllGather src operand is tiled, "
                  "AllGather requires contiguous buffers");
            }
            for (const auto& d : concrete.dst_ins) {
              ASSERT_VALID_RUNTIME(
                  !chunk_map.contains(d),
                  "RegisterTiling: AllGather dst_in operand is tiled, "
                  "AllGather requires contiguous buffers");
            }
          }

          rw.CloneOp(i);
        },
        op.op);
  }

  return rw.Finish();
}

}  // namespace setu::planner::passes
