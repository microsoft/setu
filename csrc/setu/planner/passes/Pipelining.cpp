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
#include "planner/passes/Pipelining.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/TorchCommon.h"
#include "planner/hints/Hint.h"
#include "planner/ir/cir/Analysis.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================

namespace {

//==============================================================================
// Shared helpers
//==============================================================================

/// Trace a value back through SliceOp/ConsumeOp chains to the root defining
/// operation.  Returns the root value (defined by a non-Slice, non-Consume op).
cir::Value TraceToRoot(const cir::Program& program, cir::Value v) {
  while (true) {
    const auto& def_op = program.GetDefiningOp(v);
    if (def_op.Type() == cir::OpType::kSlice) {
      v = std::get<cir::SliceOp>(def_op.op).src;
    } else if (def_op.Type() == cir::OpType::kConsume) {
      v = std::get<cir::ConsumeOp>(def_op.op).src;
    } else {
      return v;
    }
  }
}

/// Follow a value's uses through Slice/Consume to find a user of a specific
/// OpType.  Returns the op index if found, nullopt otherwise.
std::optional<std::uint32_t> FindDownstreamOp(
    const cir::Program& program, const cir::DefUseChains& def_use,
    cir::Value start, cir::OpType target_type) {
  std::vector<cir::Value> worklist = {start};
  while (!worklist.empty()) {
    cir::Value val = worklist.back();
    worklist.pop_back();

    for (std::uint32_t user_idx : def_use.uses[val.id]) {
      const auto& user_op = program.Operations()[user_idx];
      if (user_op.Type() == cir::OpType::kSlice) {
        worklist.push_back(std::get<cir::SliceOp>(user_op.op).out);
      } else if (user_op.Type() == cir::OpType::kConsume) {
        worklist.push_back(std::get<cir::ConsumeOp>(user_op.op).out);
      } else if (user_op.Type() == target_type) {
        return user_idx;
      }
    }
  }
  return std::nullopt;
}

//==============================================================================
// CopyChain detection (existing relay-chain pipelining)
//==============================================================================

/// One hop in a relay chain.
struct ChainHop {
  std::uint32_t op_idx;  ///< CopyOp index in the source program
  cir::Value src;        ///< Source value for this hop
  cir::Value dst_in;     ///< Destination buffer (consumed)
  cir::Value dst_out;    ///< Result of the copy
};

/// A maximal linear relay chain of >= 2 hops.
struct CopyChain {
  std::vector<ChainHop> hops;
  std::size_t payload_elements;
};

/// Build relay chains from the program.
///
/// For each CopyOp, check if its dst_out feeds (through Slice/Consume) into
/// exactly one downstream CopyOp as src.  Build maximal chains from these
/// links.
std::vector<CopyChain> DetectChains(const cir::Program& program) {
  auto def_use = cir::DefUseChains::Build(program);

  // Build copy successor/predecessor maps.
  // copy_successor[op_idx] = op_idx of the CopyOp that uses this copy's
  // dst_out as src (through Slice/Consume chains).
  std::unordered_map<std::uint32_t, std::uint32_t> copy_successor;
  std::unordered_map<std::uint32_t, std::uint32_t> copy_predecessor;

  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    if (op.Type() != cir::OpType::kCopy) continue;

    const auto& copy_op = std::get<cir::CopyOp>(op.op);

    // Follow dst_out's uses through Slice/Consume chains to find downstream
    // CopyOps that use it as src.
    std::vector<std::uint32_t> downstream_copies;

    // BFS/DFS through uses of dst_out and any Slice/Consume outputs
    std::vector<cir::Value> worklist = {copy_op.dst_out};
    while (!worklist.empty()) {
      auto val = worklist.back();
      worklist.pop_back();

      for (auto user_op_idx : def_use.uses[val.id]) {
        const auto& user_op = program.Operations()[user_op_idx];
        if (user_op.Type() == cir::OpType::kSlice) {
          // Slice produces a new value — follow it
          worklist.push_back(std::get<cir::SliceOp>(user_op.op).out);
        } else if (user_op.Type() == cir::OpType::kConsume) {
          // Consume produces a new value — follow it
          worklist.push_back(std::get<cir::ConsumeOp>(user_op.op).out);
        } else if (user_op.Type() == cir::OpType::kCopy) {
          // Check if this copy uses val as src (not dst_in)
          const auto& user_copy = std::get<cir::CopyOp>(user_op.op);
          auto root_src = TraceToRoot(program, user_copy.src);
          if (root_src == copy_op.dst_out) {
            downstream_copies.push_back(user_op_idx);
          }
        }
      }
    }

    ASSERT_VALID_RUNTIME(
        downstream_copies.size() <= 1,
        "Pipelining: CopyOp at [{}] has {} downstream copies (expected <= 1, "
        "linearity violation)",
        op_idx, downstream_copies.size());

    if (downstream_copies.size() == 1) {
      copy_successor[op_idx] = downstream_copies[0];
      copy_predecessor[downstream_copies[0]] = op_idx;
    }
  }

  // Build chains starting from heads (copies with no predecessor).
  std::vector<CopyChain> chains;
  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    if (op.Type() != cir::OpType::kCopy) continue;
    if (copy_predecessor.contains(op_idx)) continue;  // not a chain head
    if (!copy_successor.contains(op_idx)) continue;   // single hop, no chain

    CopyChain chain;
    auto current = op_idx;
    while (true) {
      const auto& copy_op =
          std::get<cir::CopyOp>(program.Operations()[current].op);
      chain.hops.push_back(ChainHop{
          .op_idx = current,
          .src = copy_op.src,
          .dst_in = copy_op.dst_in,
          .dst_out = copy_op.dst_out,
      });
      auto it = copy_successor.find(current);
      if (it == copy_successor.end()) break;
      current = it->second;
    }

    chain.payload_elements =
        program.GetValueInfo(chain.hops[0].src).size_elements;
    chains.push_back(std::move(chain));
  }

  return chains;
}

/// Find the ConsumeOp that consumes the last hop's dst_in (the final
/// destination buffer).  Returns the op index, or nullopt if not found.
std::optional<std::uint32_t> FindChainConsumeOp(
    const cir::Program& program, const cir::DefUseChains& def_use,
    const CopyChain& chain) {
  // The consume is on dst_out (which aliases dst_in).
  auto last_dst_out = chain.hops.back().dst_out;
  for (auto user_op_idx : def_use.uses[last_dst_out.id]) {
    if (program.Operations()[user_op_idx].Type() == cir::OpType::kConsume) {
      return user_op_idx;
    }
  }
  return std::nullopt;
}

//==============================================================================
// Pack → Copy → Unpack chain detection and pipelining
//==============================================================================

/// A chunk of consecutive pieces from a Pack/Unpack to be transferred together.
struct PieceChunk {
  std::size_t first_piece;     ///< Index of the first piece in this chunk
  std::size_t num_pieces;      ///< Number of pieces in this chunk
  std::size_t total_elements;  ///< Sum of element counts across all pieces
};

/// A detected Pack → Copy → Unpack chain produced by PackUnpackCopies.
struct PackCopyUnpackChain {
  std::uint32_t alloc_src_idx;  ///< AllocTmpOp for pack's dst_in
  std::uint32_t pack_idx;       ///< PackOp
  std::uint32_t alloc_dst_idx;  ///< AllocTmpOp for copy's dst_in
  std::uint32_t copy_idx;       ///< CopyOp (cross-device)
  std::uint32_t unpack_idx;     ///< UnpackOp
  std::size_t total_elements;   ///< Total payload in elements
  torch::Dtype dtype;
};

/// Split pieces into chunks where each chunk's total element count does not
/// exceed chunk_size_elements.  Pieces are kept whole (never split).
/// Returns empty vector if fewer than 2 chunks would result.
std::vector<PieceChunk> SplitPiecesIntoChunks(
    const cir::Program& program,
    const std::vector<cir::Value>& pieces,
    std::size_t chunk_size_elements) {
  std::vector<PieceChunk> chunks;
  std::size_t piece_idx = 0;

  while (piece_idx < pieces.size()) {
    PieceChunk chunk{piece_idx, 0, 0};
    while (piece_idx < pieces.size()) {
      std::size_t piece_size =
          program.GetValueInfo(pieces[piece_idx]).size_elements;
      if (chunk.num_pieces > 0 &&
          chunk.total_elements + piece_size > chunk_size_elements) {
        break;
      }
      chunk.num_pieces++;
      chunk.total_elements += piece_size;
      piece_idx++;
    }
    chunks.push_back(chunk);
  }

  if (chunks.size() < 2) return {};
  return chunks;
}

/// Detect Pack → Copy → Unpack chains in the program.
///
/// Pattern: PackOp.dst_out → CopyOp.src, CopyOp.dst_out → UnpackOp.src.
/// Also traces PackOp.dst_in and CopyOp.dst_in back to their AllocTmpOps.
std::vector<PackCopyUnpackChain> DetectPackCopyUnpackChains(
    const cir::Program& program) {
  auto def_use = cir::DefUseChains::Build(program);
  std::vector<PackCopyUnpackChain> chains;

  for (std::uint32_t op_idx = 0; op_idx < program.NumOperations(); ++op_idx) {
    const auto& op = program.Operations()[op_idx];
    if (op.Type() != cir::OpType::kPack) continue;

    const auto& pack_op = std::get<cir::PackOp>(op.op);

    // Find downstream CopyOp from pack's dst_out.
    auto copy_idx =
        FindDownstreamOp(program, def_use, pack_op.dst_out, cir::OpType::kCopy);
    if (!copy_idx.has_value()) continue;

    const auto& copy_op =
        std::get<cir::CopyOp>(program.Operations()[*copy_idx].op);

    // Verify it's cross-device.
    auto src_device = program.GetValueInfo(copy_op.src).device;
    auto dst_device = program.GetValueInfo(copy_op.dst_in).device;
    if (src_device == dst_device) continue;

    // Find downstream UnpackOp from copy's dst_out.
    auto unpack_idx = FindDownstreamOp(program, def_use, copy_op.dst_out,
                                       cir::OpType::kUnpack);
    if (!unpack_idx.has_value()) continue;

    // Trace back to AllocTmpOps for pack.dst_in and copy.dst_in.
    const auto& pack_alloc_op = program.GetDefiningOp(pack_op.dst_in);
    if (pack_alloc_op.Type() != cir::OpType::kAllocTmp) continue;
    std::uint32_t alloc_src_idx =
        program.GetValueInfo(pack_op.dst_in).def_op_index;

    const auto& copy_alloc_op = program.GetDefiningOp(copy_op.dst_in);
    if (copy_alloc_op.Type() != cir::OpType::kAllocTmp) continue;
    std::uint32_t alloc_dst_idx =
        program.GetValueInfo(copy_op.dst_in).def_op_index;

    chains.push_back(PackCopyUnpackChain{
        .alloc_src_idx = alloc_src_idx,
        .pack_idx = op_idx,
        .alloc_dst_idx = alloc_dst_idx,
        .copy_idx = *copy_idx,
        .unpack_idx = *unpack_idx,
        .total_elements =
            std::get<cir::AllocTmpOp>(pack_alloc_op.op).size_elements,
        .dtype = std::get<cir::AllocTmpOp>(pack_alloc_op.op).dtype,
    });
  }

  return chains;
}

/// Emit pipelined Pack → Copy → Unpack wavefront for a single chain.
///
/// Splits the chain's pieces into K chunks and emits them in wavefront order
/// with 3 hops: pack (same-device), copy (cross-device), unpack (same-device).
void EmitPipelinedPackCopyUnpack(
    cir::ProgramRewriter& rw,
    const cir::Program& source,
    const PackCopyUnpackChain& chain,
    const std::vector<PieceChunk>& chunks) {
  const auto& pack_op =
      std::get<cir::PackOp>(source.Operations()[chain.pack_idx].op);
  const auto& unpack_op =
      std::get<cir::UnpackOp>(source.Operations()[chain.unpack_idx].op);

  // Device info from the original pack sources and unpack destinations.
  cir::Device src_device = source.GetValueInfo(pack_op.srcs[0]).device;
  cir::Device dst_device = source.GetValueInfo(unpack_op.dst_ins[0]).device;

  std::size_t num_chunks = chunks.size();
  constexpr std::size_t kNumHops = 3;
  std::size_t num_micro_stages = num_chunks + kNumHops - 1;

  // Per-chunk intermediate values produced by pack (hop 0) and copy (hop 1).
  std::vector<cir::Value> packed(num_chunks);
  std::vector<cir::Value> copied(num_chunks);

  LOG_DEBUG(
      "Pipelining: Pack-Copy-Unpack wavefront with {} chunks, "
      "{} micro-stages, {} total pieces",
      num_chunks, num_micro_stages, pack_op.srcs.size());

  for (std::size_t s = 0; s < num_micro_stages; ++s) {
    std::size_t hop_start =
        (s < num_chunks) ? std::size_t{0} : s - num_chunks + 1;
    std::size_t hop_end = std::min(s + 1, kNumHops);

    // Iterate hops in reverse within each diagonal (same convention as
    // CopyChain wavefront).
    for (std::size_t hop_idx = hop_end; hop_idx-- > hop_start;) {
      std::size_t chunk_idx = s - hop_idx;
      if (chunk_idx >= num_chunks) continue;
      const auto& chunk = chunks[chunk_idx];

      if (hop_idx == 0) {
        // Pack: gather this chunk's pieces into a contiguous src temp.
        std::vector<cir::Value> chunk_srcs;
        chunk_srcs.reserve(chunk.num_pieces);
        for (std::size_t p = 0; p < chunk.num_pieces; ++p) {
          chunk_srcs.push_back(
              rw.Lookup(pack_op.srcs[chunk.first_piece + p]));
        }
        cir::Value src_tmp = rw.Target().EmitAllocTmp(
            src_device, chunk.total_elements, chain.dtype);
        packed[chunk_idx] = rw.Target().EmitPack(chunk_srcs, src_tmp);

      } else if (hop_idx == 1) {
        // Copy: cross-device transfer of this chunk's packed buffer.
        cir::Value dst_tmp = rw.Target().EmitAllocTmp(
            dst_device, chunk.total_elements, chain.dtype);
        copied[chunk_idx] =
            rw.Target().EmitCopy(packed[chunk_idx], dst_tmp);

      } else {
        // Unpack: scatter this chunk from dst temp to destination pieces.
        std::vector<cir::Value> chunk_dst_ins;
        chunk_dst_ins.reserve(chunk.num_pieces);
        for (std::size_t p = 0; p < chunk.num_pieces; ++p) {
          chunk_dst_ins.push_back(
              rw.Lookup(unpack_op.dst_ins[chunk.first_piece + p]));
        }
        std::vector<cir::Value> unpack_results =
            rw.Target().EmitUnpack(copied[chunk_idx], chunk_dst_ins);

        for (std::size_t j = 0; j < chunk.num_pieces; ++j) {
          rw.MapValue(unpack_op.dst_outs[chunk.first_piece + j],
                      unpack_results[j]);
        }
      }
    }
  }
}

//==============================================================================
}  // namespace
//==============================================================================

cir::Program Pipelining::Run(cir::Program program, const PassContext& ctx) {
  // Allow per-operation hint to override the constructor chunk size.
  auto chunk_size_bytes = chunk_size_bytes_;
  auto hint_refs = ctx.hints.GetHints<hints::PipelineChunkSizeHint>();
  if (!hint_refs.empty()) {
    chunk_size_bytes = hint_refs.front().get().chunk_size_bytes;
    LOG_DEBUG("Pipelining: using chunk_size_bytes={} from hint",
              chunk_size_bytes);
  }

  // ---- Detect CopyChains (existing relay-chain pipelining) ----

  auto chains = DetectChains(program);

  std::erase_if(chains, [&](const CopyChain& chain) {
    if (chain.hops.size() < 2) return true;
    auto dtype = program.GetValueInfo(chain.hops[0].src).dtype;
    auto element_size = static_cast<std::size_t>(torch::elementSize(dtype));
    auto chunk_size_elements = chunk_size_bytes / element_size;
    return chain.payload_elements <= chunk_size_elements;
  });

  // ---- Detect Pack → Copy → Unpack chains ----

  auto pcu_chains = DetectPackCopyUnpackChains(program);

  // Filter and split into chunks; drop chains that don't need pipelining.
  std::vector<std::vector<PieceChunk>> pcu_chunks_vec;
  std::erase_if(pcu_chains, [&](const PackCopyUnpackChain& pcu) {
    auto element_size =
        static_cast<std::size_t>(torch::elementSize(pcu.dtype));
    auto chunk_size_elements = chunk_size_bytes / element_size;
    if (pcu.total_elements <= chunk_size_elements) return true;

    const auto& pack_op =
        std::get<cir::PackOp>(program.Operations()[pcu.pack_idx].op);
    auto chunks = SplitPiecesIntoChunks(program, pack_op.srcs,
                                        chunk_size_elements);
    if (chunks.empty()) return true;

    pcu_chunks_vec.push_back(std::move(chunks));
    return false;
  });

  // ---- Early exit if nothing to pipeline ----

  if (chains.empty() && pcu_chains.empty()) {
    return program;
  }

  // ---- Build lookup structures ----

  auto def_use = cir::DefUseChains::Build(program);

  // Map from op_idx → chain index for CopyChain heads
  std::unordered_map<std::uint32_t, std::size_t> chain_head_map;
  // Map from op_idx → PCU chain index for PackCopyUnpack heads
  std::unordered_map<std::uint32_t, std::size_t> pcu_head_map;
  // Set of op indices to skip
  std::unordered_set<std::uint32_t> skip_ops;

  for (std::size_t ci = 0; ci < chains.size(); ++ci) {
    const auto& chain = chains[ci];
    chain_head_map[chain.hops[0].op_idx] = ci;

    // Mark mid-chain copies for skipping
    for (std::size_t hi = 1; hi < chain.hops.size(); ++hi) {
      skip_ops.insert(chain.hops[hi].op_idx);
    }

    // Mark the chain's ConsumeOp for skipping (we emit it ourselves)
    auto consume_idx = FindChainConsumeOp(program, def_use, chain);
    if (consume_idx.has_value()) {
      skip_ops.insert(*consume_idx);
    }
  }

  for (std::size_t pi = 0; pi < pcu_chains.size(); ++pi) {
    const auto& pcu = pcu_chains[pi];
    pcu_head_map[pcu.pack_idx] = pi;

    // Skip all ops in the chain except the head (pack_idx).
    // The head triggers wavefront emission via pcu_head_map.
    skip_ops.insert(pcu.alloc_src_idx);
    skip_ops.insert(pcu.alloc_dst_idx);
    skip_ops.insert(pcu.copy_idx);
    skip_ops.insert(pcu.unpack_idx);
  }

  // ---- Rewrite ----

  auto rw = cir::ProgramRewriter(program);

  for (std::size_t i = 0; i < program.NumOperations(); ++i) {
    auto idx = static_cast<std::uint32_t>(i);

    if (skip_ops.contains(idx)) {
      continue;
    }

    // Check if this is a CopyChain head.
    auto head_it = chain_head_map.find(idx);
    if (head_it != chain_head_map.end()) {
      // Emit CopyChain wavefront (existing logic).
      const auto& chain = chains[head_it->second];
      auto num_hops = chain.hops.size();
      auto payload = chain.payload_elements;
      auto dtype = program.GetValueInfo(chain.hops[0].src).dtype;
      auto element_size = static_cast<std::size_t>(torch::elementSize(dtype));
      auto chunk_size = chunk_size_bytes / element_size;
      auto num_chunks = (payload + chunk_size - 1) / chunk_size;
      auto num_micro_stages = num_chunks + num_hops - 1;

      std::vector<cir::Value> prev_dst_out(num_chunks);

      for (std::size_t s = 0; s < num_micro_stages; ++s) {
        auto hop_start =
            (s < num_chunks) ? std::size_t{0} : s - num_chunks + 1;
        auto hop_end = std::min(s + 1, num_hops);

        for (auto hop_idx = hop_end; hop_idx-- > hop_start;) {
          auto chunk_idx = s - hop_idx;
          ASSERT_VALID_RUNTIME(chunk_idx < num_chunks,
                               "chunk_idx {} out of range", chunk_idx);

          auto chunk_offset = chunk_idx * chunk_size;
          auto chunk_size_actual =
              std::min(chunk_size, payload - chunk_offset);
          auto slice_spec = cir::Slice{chunk_offset, chunk_size_actual};

          cir::Value src_chunk;
          if (hop_idx == 0) {
            src_chunk = rw.Target().EmitSlice(
                rw.Lookup(chain.hops[0].src), slice_spec);
          } else {
            src_chunk = prev_dst_out[chunk_idx];
          }

          auto dst_chunk = rw.Target().EmitSlice(
              rw.Lookup(chain.hops[hop_idx].dst_in), slice_spec);

          prev_dst_out[chunk_idx] =
              rw.Target().EmitCopy(src_chunk, dst_chunk);
        }
      }

      auto new_final =
          rw.Target().EmitConsume(rw.Lookup(chain.hops.back().dst_in));
      rw.MapValue(chain.hops.back().dst_out, new_final);
      continue;
    }

    // Check if this is a PackCopyUnpack chain head.
    auto pcu_it = pcu_head_map.find(idx);
    if (pcu_it != pcu_head_map.end()) {
      EmitPipelinedPackCopyUnpack(rw, program, pcu_chains[pcu_it->second],
                                  pcu_chunks_vec[pcu_it->second]);
      continue;
    }

    // Default: clone the operation.
    rw.CloneOp(i);
  }

  return rw.Finish();
}

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
