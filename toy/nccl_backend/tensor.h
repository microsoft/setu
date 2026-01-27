#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <memory>

using ShardId = std::size_t;

struct TensorShardSpec {
    std::size_t start;
    std::size_t end;
    std::size_t dim;
    std::size_t device_id;
};

struct Shard {
    TensorShardSpec spec;
    float* device_ptr;
};

struct Tensor {
    std::string name;
    std::unordered_map<ShardId, Shard> shards;
};

using TensorStore = std::unordered_map<std::string, std::unique_ptr<Tensor>>;