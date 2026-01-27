#pragma once

#include "planner.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <nccl.h>
#include <cuda_runtime.h>

class Worker {
public:
    Worker(std::size_t device_id) : device_id_(device_id), running_(false), stop_requested_(false) {}

    void Start();

    bool IsAlive();

    void Stop();

    virtual void Setup() {}

    virtual bool Execute(Program program) = 0;

    void EnqueueProgram(Program program);

protected:
    std::size_t device_id_;

private:
    void WorkerLoop();

    std::thread worker_thread_;
    std::queue<Program> program_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
};

class NCCLWorker : public Worker {
public:
    NCCLWorker(std::size_t device_id);
    ~NCCLWorker();

    void Setup() override;

    bool Execute(Program program) override;

    void SetComm(ncclComm_t comm) { comm_ = comm; }
    void SetStream(cudaStream_t stream) { stream_ = stream; }

private:

using CommCacheKey = std::string;
struct CommCacheEntry {
    ncclComm_t nccl_comm;
    std::unordered_map<DeviceId, int> device_to_rank;
};

    ncclComm_t comm_;
    cudaStream_t stream_;
    std::unordered_map<CommCacheKey, CommCacheEntry> comm_cache_;
    CommCacheKey active_comm_key_;

    std::string CommIdToString(const ncclUniqueId& id);
};