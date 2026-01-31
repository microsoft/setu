#include "node_agent.h"
#include "cuda_helpers.h"
#include <utility>
#include <stdexcept>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

NodeAgent::NodeAgent(PlannerPtr planner)
    : planner_(std::move(planner)) {}

NodeAgent::~NodeAgent() {
    for (auto& [device_id, worker] : workers_) {
        worker->Stop();
    }
}

void NodeAgent::PerformCopy(CopySpec& copy_spec) {
    auto plan = planner_->Compile(copy_spec, store_);
    for (auto [device_id, program]: plan.device_programs) {
        workers_.at(device_id)->EnqueueProgram(program);
    }
}

ShardId NodeAgent::RegisterTensorShard(std::string tensor_name, TensorShardSpec info) {
    spdlog::debug("RegisterTensorShard: tensor='{}', device={}, range=[{}, {})",
                  tensor_name, info.device_id, info.start, info.end);

    ValidateShard(tensor_name, info);
    EnsureWorkerIsReady(info.device_id);

    auto it = store_.find(tensor_name);
    ShardId shard_id;

    std::size_t shard_size = info.end - info.start;
    std::size_t bytes = shard_size * sizeof(float);
    float* device_ptr = nullptr;

    CUDACHECK(cudaSetDevice(info.device_id));
    CUDACHECK(cudaMalloc(&device_ptr, bytes));
    spdlog::debug("RegisterTensorShard: allocated {} bytes, device_ptr={}", bytes, (void*)device_ptr);

    if (it == store_.end()) {
        auto tensor = std::make_unique<Tensor>();
        tensor->name = tensor_name;
        shard_id = 0;

        Shard shard;
        shard.spec = info;
        shard.device_ptr = device_ptr;
        tensor->shards[shard_id] = shard;

        store_[tensor_name] = std::move(tensor);
        spdlog::debug("RegisterTensorShard: created new tensor '{}', shard_id={}", tensor_name, shard_id);
    } else {
        shard_id = it->second->shards.size();

        Shard shard;
        shard.spec = info;
        shard.device_ptr = device_ptr;
        it->second->shards[shard_id] = shard;

        spdlog::debug("RegisterTensorShard: added to existing tensor '{}', shard_id={}", tensor_name, shard_id);
    }

    return shard_id;
}

bool NodeAgent::IsTensorRegistered(std::string tensor_name) {
    auto it = store_.find(tensor_name);
    if (it == store_.end() || it->second->shards.empty()) {
        return false;
    }

    const auto& shards = it->second->shards;
    std::size_t dim = shards.begin()->second.spec.dim;

    std::size_t total_elements = 0;
    for (const auto& [shard_id, shard] : shards) {
        total_elements += (shard.spec.end - shard.spec.start);
    }

    return total_elements == dim;
}

void NodeAgent::ValidateShard(const std::string& tensor_name, const TensorShardSpec& info) {
    if (info.start >= info.end) {
        throw std::invalid_argument("Shard start must be less than end");
    }

    if (info.end > info.dim) {
        throw std::invalid_argument("Shard end cannot exceed tensor dimension");
    }

    auto it = store_.find(tensor_name);
    if (it != store_.end()) {
        for (const auto& [shard_id, shard] : it->second->shards) {
            if (shard.spec.dim != info.dim) {
                throw std::invalid_argument("All shards must have the same dimension");
            }

            bool overlap = !(info.end <= shard.spec.start || info.start >= shard.spec.end);
            if (overlap) {
                throw std::invalid_argument("Shards cannot overlap");
            }
        }
    }
}

Tensor& NodeAgent::GetTensor(std::string tensor_name) {
    auto it = store_.find(tensor_name);
    if (it == store_.end()) {
        throw std::invalid_argument("Tensor not found");
    }

    return *it->second;
}

void NodeAgent::EnsureWorkerIsReady(DeviceId device_id) {
    auto it = workers_.find(device_id);

    if (it == workers_.end()) {
        auto worker = std::make_unique<NCCLWorker>(device_id);
        worker->Start();
        workers_[device_id] = std::move(worker);
    } else if (!it->second->IsAlive()) {
        it->second->Start();
    }
}