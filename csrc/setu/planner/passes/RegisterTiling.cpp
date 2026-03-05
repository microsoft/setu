#include "planner/passes/RegisterTiling.h"

#include "commons/Logging.h"

namespace setu::planner::passes {

using ChunkMap = std::unordered_map<cir::Value, std::vector<cir::Value>>;

/// Number of elements that fit in one register-sized chunk for a given dtype.
inline std::size_t NumElementsInChunk(std::size_t chunk_size_bytes,
                                      torch::Dtype dtype) {
  auto element_size = torch::elementSize(dtype);
  ASSERT_VALID_RUNTIME(chunk_size_bytes >= element_size,
                       "Element size {} exceeds chunk size {}", element_size,
                       chunk_size_bytes);
  return chunk_size_bytes / element_size;
}

cir::Program RegisterTiling::Run(cir::Program program,
                                 const HintStore& /*hints*/) {
  bool has_large_tmp = false;
  for (const auto& op : program.Operations()) {
    if (op.Type() == cir::OpType::kAllocTmp) {
      const auto& alloc = std::get<cir::AllocTmpOp>(op.op);
      if (alloc.size_elements >
          NumElementsInChunk(chunk_size_bytes_, alloc.dtype)) {
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
                NumElementsInChunk(chunk_size_bytes_, concrete.dtype);

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

          rw.CloneOp(i);
        },
        op.op);
  }

  return rw.Finish();
}

}  // namespace setu::planner::passes
