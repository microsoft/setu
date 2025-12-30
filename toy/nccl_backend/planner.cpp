#include "planner.h"
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iomanip>

std::string NCCLPlanner::CommIdToString(const ncclUniqueId& id) {
    std::stringstream ss;
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0')
           << (int)(unsigned char)id.internal[i];
    }
    return ss.str();
}

Plan NCCLPlanner::Compile(CopySpec& copy_spec, const TensorStore& tensor_store) {
    const auto& src_tensor = tensor_store.at(copy_spec.src_tensor_name);
    const auto& dst_tensor = tensor_store.at(copy_spec.dst_tensor_name);

    if (src_tensor->shards.empty() || dst_tensor->shards.empty()) {
        throw std::invalid_argument("Tensors must have shards");
    }

    std::size_t src_dim = src_tensor->shards.begin()->second.spec.dim;
    std::size_t dst_dim = dst_tensor->shards.begin()->second.spec.dim;

    if (src_dim != dst_dim) {
        throw std::invalid_argument("Source and destination tensor sizes do not match");
    }

    std::vector<std::pair<ShardId, const Shard*>> src_shards;
    for (const auto& [id, shard] : src_tensor->shards) {
        src_shards.push_back({id, &shard});
    }
    std::sort(src_shards.begin(), src_shards.end(),
              [](const auto& a, const auto& b) { return a.second->spec.start < b.second->spec.start; });

    std::vector<std::pair<ShardId, const Shard*>> dst_shards;
    for (const auto& [id, shard] : dst_tensor->shards) {
        dst_shards.push_back({id, &shard});
    }
    std::sort(dst_shards.begin(), dst_shards.end(),
              [](const auto& a, const auto& b) { return a.second->spec.start < b.second->spec.start; });

    Plan plan;
    std::set<DeviceId> participating_devices;

    std::size_t src_idx = 0;
    std::size_t dst_idx = 0;
    std::size_t src_offset = 0;
    std::size_t dst_offset = 0;

    while (src_idx < src_shards.size() && dst_idx < dst_shards.size()) {
        auto [src_shard_id, src_shard] = src_shards[src_idx];
        auto [dst_shard_id, dst_shard] = dst_shards[dst_idx];

        std::size_t src_remaining = (src_shard->spec.end - src_shard->spec.start) - src_offset;
        std::size_t dst_remaining = (dst_shard->spec.end - dst_shard->spec.start) - dst_offset;
        std::size_t to_copy = std::min(src_remaining, dst_remaining);

        DeviceId src_device = src_shard->spec.device_id;
        DeviceId dst_device = dst_shard->spec.device_id;

        if (src_device == dst_device) {
            plan.device_programs[src_device].instrs.push_back(CopyInstr{
                .src_tensor_name = copy_spec.src_tensor_name,
                .src_shard_id = src_shard_id,
                .src_handle = src_shard->device_ptr,
                .src_offset = src_offset,
                .dst_tensor_name = copy_spec.dst_tensor_name,
                .dst_shard_id = dst_shard_id,
                .dst_handle = dst_shard->device_ptr,
                .dst_offset = dst_offset,
                .count = to_copy
            });
        } else {
            participating_devices.insert(src_device);
            participating_devices.insert(dst_device);

            plan.device_programs[src_device].instrs.push_back(SendInstr{
                .dest_device = dst_device,
                .tensor_name = copy_spec.src_tensor_name,
                .shard_id = src_shard_id,
                .handle = src_shard->device_ptr,
                .offset = src_offset,
                .count = to_copy
            });

            plan.device_programs[dst_device].instrs.push_back(RecvInstr{
                .src_device = src_device,
                .tensor_name = copy_spec.dst_tensor_name,
                .shard_id = dst_shard_id,
                .handle = dst_shard->device_ptr,
                .offset = dst_offset,
                .count = to_copy
            });
        }

        src_offset += to_copy;
        dst_offset += to_copy;

        if (src_offset >= (src_shard->spec.end - src_shard->spec.start)) {
            src_idx++;
            src_offset = 0;
        }

        if (dst_offset >= (dst_shard->spec.end - dst_shard->spec.start)) {
            dst_idx++;
            dst_offset = 0;
        }
    }

    if (!participating_devices.empty()) {
        ncclUniqueId comm_id;
        bool needs_init = false;

        if (comm_cache_.find(participating_devices) != comm_cache_.end()) {
            comm_id = comm_cache_[participating_devices];
        } else {
            ncclGetUniqueId(&comm_id);
            comm_cache_[participating_devices] = comm_id;
            needs_init = true;
        }

        std::unordered_map<DeviceId, int> device_to_rank;
        int rank = 0;
        for (DeviceId device_id : participating_devices) {
            device_to_rank[device_id] = rank++;
        }

        for (DeviceId device_id : participating_devices) {
            std::vector<Instruction> new_instrs;

            if (needs_init) {
                new_instrs.push_back(InitCommInstr{
                    .comm_id = comm_id,
                    .device_to_rank = device_to_rank
                });
            } else {
                new_instrs.push_back(UseCommInstr{.comm_id = comm_id});
            }

            new_instrs.insert(new_instrs.end(), plan.device_programs[device_id].instrs.begin(), plan.device_programs[device_id].instrs.end());
            plan.device_programs[device_id].instrs = std::move(new_instrs);
        }
    }

    return plan;
}