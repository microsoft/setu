#pragma once

#include <unordered_map>
#include <memory>

#include "copy_spec.h"
#include "planner.h"
#include "tensor.h"
#include "worker.h"

class NodeAgent {
public:
    NodeAgent(PlannerPtr planner);
    ~NodeAgent();

    ShardId RegisterTensorShard(std::string tensor_name, TensorShardSpec info);

    bool IsTensorRegistered(std::string tensor_name);

    void PerformCopy(CopySpec& copy_spec);

    Tensor& GetTensor(std::string tensor_name);

    const TensorStore& GetTensorStore() const { return store_; }

private:
    void EnsureWorkerIsReady(DeviceId device_id);
    void ValidateShard(const std::string& tensor_name, const TensorShardSpec& info);

    PlannerPtr planner_;
    TensorStore store_;
    std::unordered_map<DeviceId, std::unique_ptr<Worker>> workers_;
};