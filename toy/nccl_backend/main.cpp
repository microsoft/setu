#include "node_agent.h"
#include "planner.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <chrono>

void InitLogLevel() {
    const char* log_level_env = std::getenv("LOG_LEVEL");
    if (log_level_env) {
        std::string level_str(log_level_env);
        if (level_str == "trace") {
            spdlog::set_level(spdlog::level::trace);
        } else if (level_str == "debug") {
            spdlog::set_level(spdlog::level::debug);
        } else if (level_str == "info") {
            spdlog::set_level(spdlog::level::info);
        } else if (level_str == "warn") {
            spdlog::set_level(spdlog::level::warn);
        } else if (level_str == "error") {
            spdlog::set_level(spdlog::level::err);
        } else if (level_str == "critical") {
            spdlog::set_level(spdlog::level::critical);
        } else if (level_str == "off") {
            spdlog::set_level(spdlog::level::off);
        }
    }
}

std::string CommIdToReadableString(const ncclUniqueId& id) {
    std::string full = NCCLPlanner::CommIdToString(id);
    return full.substr(0, std::min(10UL, full.size())) + "...";
}

void InitTensor(Tensor& tensor, float value) {
    for (auto& [shard_id, shard] : tensor.shards) {
        assert(shard.device_ptr != nullptr);

        std::size_t shard_size = shard.spec.end - shard.spec.start;
        std::size_t bytes = shard_size * sizeof(float);

        std::vector<float> host_data(shard_size, value);

        cudaSetDevice(shard.spec.device_id);
        cudaMemcpy(shard.device_ptr, host_data.data(), bytes, cudaMemcpyHostToDevice);
    }
}

void PrintTensorFirstElements(const Tensor& tensor, std::size_t n = 5) {
    std::cout << "Tensor '" << tensor.name << "':" << std::endl;

    std::vector<std::pair<ShardId, const Shard*>> sorted_shards;
    for (const auto& [id, shard] : tensor.shards) {
        sorted_shards.push_back({id, &shard});
    }
    std::sort(sorted_shards.begin(), sorted_shards.end(),
              [](const auto& a, const auto& b) { return a.second->spec.start < b.second->spec.start; });

    for (const auto& [shard_id, shard] : sorted_shards) {
        std::size_t shard_size = shard->spec.end - shard->spec.start;
        std::size_t elements_to_print = std::min(n, shard_size);

        std::vector<float> host_data(elements_to_print);

        cudaSetDevice(shard->spec.device_id);
        cudaMemcpy(host_data.data(), shard->device_ptr, elements_to_print * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "  Shard " << shard_id << " (device=" << shard->spec.device_id
                  << ", range=[" << shard->spec.start << ", " << shard->spec.end << ")): [";
        for (std::size_t i = 0; i < elements_to_print; ++i) {
            std::cout << host_data[i];
            if (i < elements_to_print - 1) std::cout << ", ";
        }
        if (elements_to_print < shard_size) {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;
    }
}

void PrintPlan(const Plan& plan) {
    std::cout << "\n=== Generated Plan ===" << std::endl;

    std::vector<std::pair<DeviceId, const Program*>> sorted_programs;
    for (const auto& [device_id, program] : plan.device_programs) {
        sorted_programs.push_back({device_id, &program});
    }
    std::sort(sorted_programs.begin(), sorted_programs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (const auto& [device_id, program_ptr] : sorted_programs) {
        const auto& program = *program_ptr;
        std::cout << "\nDevice " << device_id << " program:" << std::endl;
        for (const auto& instr : program.instrs) {
            if (std::holds_alternative<SendInstr>(instr)) {
                const auto& send = std::get<SendInstr>(instr);
                std::cout << "  SEND(tensor=" << send.tensor_name
                          << ", shard=" << send.shard_id
                          << ", handle=" << send.handle
                          << ", offset=" << send.offset
                          << ", to_device=" << send.dest_device
                          << ", count=" << send.count << ")" << std::endl;
            } else if (std::holds_alternative<RecvInstr>(instr)) {
                const auto& recv = std::get<RecvInstr>(instr);
                std::cout << "  RECV(tensor=" << recv.tensor_name
                          << ", shard=" << recv.shard_id
                          << ", handle=" << recv.handle
                          << ", offset=" << recv.offset
                          << ", from_device=" << recv.src_device
                          << ", count=" << recv.count << ")" << std::endl;
            } else if (std::holds_alternative<CopyInstr>(instr)) {
                const auto& copy = std::get<CopyInstr>(instr);
                std::cout << "  COPY(src_tensor=" << copy.src_tensor_name
                          << ", src_shard=" << copy.src_shard_id
                          << ", src_handle=" << copy.src_handle
                          << ", src_offset=" << copy.src_offset
                          << ", dst_tensor=" << copy.dst_tensor_name
                          << ", dst_shard=" << copy.dst_shard_id
                          << ", dst_handle=" << copy.dst_handle
                          << ", dst_offset=" << copy.dst_offset
                          << ", count=" << copy.count << ")" << std::endl;
            } else if (std::holds_alternative<InitCommInstr>(instr)) {
                const auto& init = std::get<InitCommInstr>(instr);
                int nranks = init.device_to_rank.size();
                int rank = init.device_to_rank.at(device_id);
                std::cout << "  INIT_COMM(comm_id=" << CommIdToReadableString(init.comm_id)
                          << ", nranks=" << nranks
                          << ", rank=" << rank << ")" << std::endl;
            } else if (std::holds_alternative<UseCommInstr>(instr)) {
                const auto& use = std::get<UseCommInstr>(instr);
                std::cout << "  USE_COMM(comm_id=" << CommIdToReadableString(use.comm_id) << ")" << std::endl;
            }
        }
    }
    std::cout << "======================\n" << std::endl;
}

int main() {
    InitLogLevel();

    PlannerPtr planner = std::make_unique<NCCLPlanner>();
    NodeAgent agent(std::move(planner));

    std::size_t tensor_dim = 1024;

    std::cout << "Registering tensor A shards (2 shards on devices 0 and 1)..." << std::endl;
    agent.RegisterTensorShard("A", TensorShardSpec{
        .start = 0,
        .end = 512,
        .dim = tensor_dim,
        .device_id = 0
    });
    agent.RegisterTensorShard("A", TensorShardSpec{
        .start = 512,
        .end = 1024,
        .dim = tensor_dim,
        .device_id = 1
    });

    // std::cout << "Registering tensor B shards (4 shards on devices 0,1,2,3)..." << std::endl;
    // agent.RegisterTensorShard("B", TensorShardSpec{
    //     .start = 0,
    //     .end = 256,
    //     .dim = tensor_dim,
    //     .device_id = 0
    // });
    // agent.RegisterTensorShard("B", TensorShardSpec{
    //     .start = 256,
    //     .end = 512,
    //     .dim = tensor_dim,
    //     .device_id = 1
    // });
    // agent.RegisterTensorShard("B", TensorShardSpec{
    //     .start = 512,
    //     .end = 768,
    //     .dim = tensor_dim,
    //     .device_id = 2
    // });
    // agent.RegisterTensorShard("B", TensorShardSpec{
    //     .start = 768,
    //     .end = 1024,
    //     .dim = tensor_dim,
    //     .device_id = 3
    // });

    std::cout << "Registering tensor C shard (1 shard on device 3)..." << std::endl;
    agent.RegisterTensorShard("C", TensorShardSpec{
        .start = 0,
        .end = 1024,
        .dim = tensor_dim,
        .device_id = 3
    });

    std::cout << "Tensor A registered: " << agent.IsTensorRegistered("A") << std::endl;
    std::cout << "Tensor B registered: " << agent.IsTensorRegistered("B") << std::endl;
    std::cout << "Tensor C registered: " << agent.IsTensorRegistered("C") << std::endl;

    std::cout << "\nInitializing tensors..." << std::endl;
    InitTensor(agent.GetTensor("A"), 10.0f);
    // InitTensor(agent.GetTensor("B"), 0.0f);
    InitTensor(agent.GetTensor("C"), 0.0f);
    std::cout << "Tensors initialized: A=10.0, B=0.0, C=0.0" << std::endl;

    // std::cout << "\nCompiling copy(A, B) plan..." << std::endl;
    // CopySpec copy_ab{
    //     .src_tensor_name = "A",
    //     .dst_tensor_name = "B"
    // };
    // auto plan_ab = planner->Compile(copy_ab, agent.GetTensorStore());
    // PrintPlan(plan_ab);

    std::cout << "\nCompiling copy(A, C) plan..." << std::endl;
    PlannerPtr temp_planner = std::make_unique<NCCLPlanner>();
    CopySpec copy_ac{
        .src_tensor_name = "A",
        .dst_tensor_name = "C"
    };
    auto plan_ac = temp_planner->Compile(copy_ac, agent.GetTensorStore());
    PrintPlan(plan_ac);

    std::cout << "Performing copy(A, C) plan..." << std::endl;
    agent.PerformCopy(copy_ac);

    std::this_thread::sleep_for(std::chrono::seconds(2));

    PrintTensorFirstElements(agent.GetTensor("A"));
    PrintTensorFirstElements(agent.GetTensor("C"));


    return 0;
}
