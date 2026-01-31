#pragma once

#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <variant>
#include <set>
#include <map>
#include <nccl.h>
#include "copy_spec.h"
#include "tensor.h"

using DeviceId = std::size_t;

struct SendInstr {
    DeviceId dest_device;
    std::string tensor_name;
    ShardId shard_id;
    float* handle;
    std::size_t offset;
    std::size_t count;
};

struct RecvInstr {
    DeviceId src_device;
    std::string tensor_name;
    ShardId shard_id;
    float* handle;
    std::size_t offset;
    std::size_t count;
};

struct CopyInstr {
    std::string src_tensor_name;
    ShardId src_shard_id;
    float* src_handle;
    std::size_t src_offset;
    std::string dst_tensor_name;
    ShardId dst_shard_id;
    float* dst_handle;
    std::size_t dst_offset;
    std::size_t count;
};

struct InitCommInstr {
    ncclUniqueId comm_id;
    std::unordered_map<DeviceId, int> device_to_rank;
};

struct UseCommInstr {
    ncclUniqueId comm_id;
};

using Instruction = std::variant<SendInstr, RecvInstr, CopyInstr, InitCommInstr, UseCommInstr>;

struct Program {
    std::vector<DeviceId> participating_devices;
    std::vector<Instruction> instrs;
};

struct Plan {
    std::unordered_map<DeviceId, Program> device_programs;
};

class Planner {
public:
    virtual ~Planner() = default;
    virtual Plan Compile(CopySpec& copy_spec, const TensorStore& tensor_store) = 0;
};

class NCCLPlanner : public Planner {
public:
    Plan Compile(CopySpec& copy_spec, const TensorStore& tensor_store) override;

    static std::string CommIdToString(const ncclUniqueId& id);

private:
    std::map<std::set<DeviceId>, ncclUniqueId> comm_cache_;
};

using PlannerPtr = std::unique_ptr<Planner>;