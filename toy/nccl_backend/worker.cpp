#include "worker.h"
#include "node_agent.h"
#include "cuda_helpers.h"
#include <vector>
#include <spdlog/spdlog.h>

NCCLWorker::NCCLWorker(std::size_t device_id)
    : Worker(device_id), comm_(nullptr), stream_(nullptr), active_comm_key_("") {
}

NCCLWorker::~NCCLWorker() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

void NCCLWorker::Setup() {
    cudaSetDevice(device_id_);
    cudaStreamCreate(&stream_);
    spdlog::debug("NCCLWorker::Setup: device={}", device_id_);
}

bool NCCLWorker::Execute(Program program) {
    spdlog::debug("Worker::Execute: device={}, num_instrs={}", device_id_, program.instrs.size());
    bool group_started = false;

    for (const auto& instr : program.instrs) {
        if (std::holds_alternative<CopyInstr>(instr)) {
            const auto& copy = std::get<CopyInstr>(instr);
            std::size_t bytes = copy.count * sizeof(float);
            cudaMemcpy(copy.dst_handle + copy.dst_offset, copy.src_handle + copy.src_offset, bytes, cudaMemcpyDeviceToDevice);
        }
        else if (std::holds_alternative<InitCommInstr>(instr)) {
            const auto& init = std::get<InitCommInstr>(instr);
            std::string key = CommIdToString(init.comm_id);

            if (comm_cache_.find(key) == comm_cache_.end()) {
                int nranks = init.device_to_rank.size();
                int rank = init.device_to_rank.at(device_id_);
                ncclComm_t comm;
                NCCLCHECK(ncclCommInitRank(&comm, nranks, init.comm_id, rank));
                comm_cache_[key] = CommCacheEntry{
                    .nccl_comm = comm,
                    .device_to_rank = init.device_to_rank
                };
            }

            active_comm_key_ = key;
        }
        else if (std::holds_alternative<UseCommInstr>(instr)) {
            const auto& use = std::get<UseCommInstr>(instr);
            active_comm_key_ = CommIdToString(use.comm_id);
        }
        else if (std::holds_alternative<SendInstr>(instr)) {
            if (!group_started) {
                ncclGroupStart();
                group_started = true;
            }
            const auto& send = std::get<SendInstr>(instr);
            const auto& entry = comm_cache_.at(active_comm_key_);
            int peer_rank = entry.device_to_rank.at(send.dest_device);
            NCCLCHECK(ncclSend(send.handle + send.offset, send.count, ncclFloat, peer_rank, entry.nccl_comm, stream_));
        }
        else if (std::holds_alternative<RecvInstr>(instr)) {
            if (!group_started) {
                ncclGroupStart();
                group_started = true;
            }
            const auto& recv = std::get<RecvInstr>(instr);
            const auto& entry = comm_cache_.at(active_comm_key_);
            int peer_rank = entry.device_to_rank.at(recv.src_device);
            NCCLCHECK(ncclRecv(recv.handle + recv.offset, recv.count, ncclFloat, peer_rank, entry.nccl_comm, stream_));
        }
    }

    if (group_started) {
        ncclGroupEnd();
        cudaStreamSynchronize(stream_);
    }

    return true;
}

std::string NCCLWorker::CommIdToString(const ncclUniqueId& id) {
    return std::string(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}

void Worker::Start() {
    if (running_) {
        return;
    }

    spdlog::debug("Worker::Start: device={}", device_id_);
    stop_requested_ = false;
    running_ = true;
    worker_thread_ = std::thread(&Worker::WorkerLoop, this);
}

void Worker::Stop() {
    if (!running_) {
        return;
    }

    spdlog::debug("Worker::Stop: device={}", device_id_);
    stop_requested_ = true;
    queue_cv_.notify_all();

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    running_ = false;
}

bool Worker::IsAlive() {
    return running_;
}

void Worker::EnqueueProgram(Program program) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        program_queue_.push(std::move(program));
    }
    queue_cv_.notify_one();
}

void Worker::WorkerLoop() {
    Setup();

    while (!stop_requested_) {
        Program program;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return stop_requested_ || !program_queue_.empty();
            });

            if (stop_requested_ && program_queue_.empty()) {
                break;
            }

            if (!program_queue_.empty()) {
                program = std::move(program_queue_.front());
                program_queue_.pop();
            } else {
                continue;
            }
        }

        Execute(program);
    }
}