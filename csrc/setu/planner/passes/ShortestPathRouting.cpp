#include "planner/passes/ShortestPathRouting.h"

#include "planner/Constants.h"

namespace setu::planner::passes {

using setu::planner::topo::Link;

cir::Program ShortestPathRouting::Run(const cir::Program& program) {
  auto rw = cir::ProgramRewriter(program);
  for (std::size_t i = 0; i < program.NumOperations(); ++i) {
    const auto& op = program.Operations()[i];
    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;
          if constexpr (std::is_same_v<T, cir::CopyOp>) {
            auto src_val_info = program.GetValueInfo(concrete.src);
            auto dst_val_info = program.GetValueInfo(concrete.dst_in);
            // builder guarantees that num bytes is the same for src, dst values
            auto bytes = src_val_info.NumBytes();

            auto path_opt =
                topo_->ShortestPath(src_val_info.device, dst_val_info.device,
                                    [bytes](const Link& l) -> float {
                                      return l.TransferTimeUs(bytes);
                                    });
            ASSERT_VALID_RUNTIME(path_opt.has_value(),
                                 "No path exists between {} and {}",
                                 src_val_info.device, dst_val_info.device);

            auto path = path_opt.value();

            // shortest path is the same as direct transfer
            // no additional hops required
            if (path.hops.size() == 2) {
              rw.CloneOp(i);
              return;
            }

            auto num_elements = src_val_info.size_elements;
            auto dt = src_val_info.dtype;
            auto element_size = torch::elementSize(dt);
            auto tmp_buf_size_elements = kRegisterSize / element_size;
            ASSERT_VALID_RUNTIME(tmp_buf_size_elements > 0,
                                 "Element size {} exceeds register size {}",
                                 element_size, kRegisterSize);

            // Allocate fixed-size temporary registers at intermediate hops
            std::vector<cir::Value> tmps;
            tmps.reserve(path.hops.size() - 2);
            for (const auto& hop :
                 std::span(path.hops.begin() + 1, path.hops.end() - 1)) {
              tmps.emplace_back(
                  rw.Target().EmitAllocTmp(hop, tmp_buf_size_elements, dt));
            }

            auto src = rw.Lookup(concrete.src);
            auto dst_in = rw.Lookup(concrete.dst_in);

            // Pipeline the payload through intermediate hops in chunks
            for (std::size_t pp_start = 0; pp_start < num_elements;
                 pp_start += tmp_buf_size_elements) {
              auto chunk_size =
                  std::min(tmp_buf_size_elements, num_elements - pp_start);

              auto src_chunk =
                  rw.Target().EmitSlice(src, cir::Slice{pp_start, chunk_size});
              auto dst_chunk = rw.Target().EmitSlice(
                  dst_in, cir::Slice{pp_start, chunk_size});

              cir::Value prev = src_chunk;
              for (std::size_t j = 0; j < tmps.size(); ++j) {
                // Slice tmp to match chunk size when payload doesn't fill
                // the full register
                auto tmp_dst = (chunk_size < tmp_buf_size_elements)
                                   ? rw.Target().EmitSlice(
                                         tmps[j], cir::Slice{0, chunk_size})
                                   : tmps[j];
                auto tmp_out = rw.Target().EmitCopy(prev, tmp_dst);
                prev = tmp_out;
              }
              (void)rw.Target().EmitCopy(prev, dst_chunk);
            }

            auto new_dst_out = rw.Target().EmitConsume(dst_in);
            rw.MapValue(concrete.dst_out, new_dst_out);

            return;
          }

          // fallthrough
          rw.CloneOp(i);
        },
        op.op);
  }
  return rw.Finish();
}
}  // namespace setu::planner::passes
