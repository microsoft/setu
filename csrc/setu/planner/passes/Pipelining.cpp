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
#include "planner/ir/cir/Analysis.h"
//==============================================================================
namespace setu::planner::passes {
//==============================================================================

namespace {

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

}  // namespace

//==============================================================================

cir::Program Pipelining::Run(cir::Program program, const PassContext& /*ctx*/) {
  auto chains = DetectChains(program);

  // Filter to chains that need pipelining
  std::erase_if(chains, [this](const CopyChain& chain) {
    return chain.hops.size() < 2 ||
           chain.payload_elements <= chunk_size_elements_;
  });

  if (chains.empty()) {
    return program;
  }

  // Build lookup structures
  auto def_use = cir::DefUseChains::Build(program);

  // Map from op_idx → chain index for chain heads
  std::unordered_map<std::uint32_t, std::size_t> chain_head_map;
  // Set of op indices to skip (mid-chain copies + consume ops)
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

  // Rewrite
  auto rw = cir::ProgramRewriter(program);

  for (std::size_t i = 0; i < program.NumOperations(); ++i) {
    if (skip_ops.contains(static_cast<std::uint32_t>(i))) {
      continue;
    }

    auto head_it = chain_head_map.find(static_cast<std::uint32_t>(i));
    if (head_it == chain_head_map.end()) {
      rw.CloneOp(i);
      continue;
    }

    // Emit wavefront for this chain
    const auto& chain = chains[head_it->second];
    auto num_hops = chain.hops.size();
    auto payload = chain.payload_elements;
    auto chunk_size = chunk_size_elements_;
    auto num_chunks = (payload + chunk_size - 1) / chunk_size;
    auto num_micro_stages = num_chunks + num_hops - 1;

    // Track previous hop's dst_out per chunk
    std::vector<cir::Value> prev_dst_out(num_chunks);

    for (std::size_t s = 0; s < num_micro_stages; ++s) {
      // Iterate (chunk_idx, hop_idx) on the wavefront diagonal
      // chunk_idx + hop_idx == s, both >= 0 and within bounds
      auto hop_start = (s < num_chunks) ? std::size_t{0} : s - num_chunks + 1;
      auto hop_end = std::min(s + 1, num_hops);

      for (auto hop_idx = hop_end; hop_idx-- > hop_start;) {
        auto chunk_idx = s - hop_idx;
        ASSERT_VALID_RUNTIME(chunk_idx < num_chunks,
                             "chunk_idx {} out of range", chunk_idx);

        auto chunk_offset = chunk_idx * chunk_size;
        auto chunk_size_actual = std::min(chunk_size, payload - chunk_offset);
        auto slice_spec = cir::Slice{chunk_offset, chunk_size_actual};

        // Source
        cir::Value src_chunk;
        if (hop_idx == 0) {
          src_chunk =
              rw.Target().EmitSlice(rw.Lookup(chain.hops[0].src), slice_spec);
        } else {
          src_chunk = prev_dst_out[chunk_idx];
        }

        // Destination
        auto dst_chunk = rw.Target().EmitSlice(
            rw.Lookup(chain.hops[hop_idx].dst_in), slice_spec);

        prev_dst_out[chunk_idx] = rw.Target().EmitCopy(src_chunk, dst_chunk);
      }
    }

    // Emit Consume for the chain's final destination and map dst_out
    auto new_final =
        rw.Target().EmitConsume(rw.Lookup(chain.hops.back().dst_in));
    rw.MapValue(chain.hops.back().dst_out, new_final);
  }

  return rw.Finish();
}

//==============================================================================
}  // namespace setu::planner::passes
//==============================================================================
