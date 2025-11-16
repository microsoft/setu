//==============================================================================
// Copyright 2025 Vajra Team; Georgia Institute of Technology; Microsoft
// Corporation
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
#include "commons/utils/TorchTensorIPC.h"
//==============================================================================
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <cstdint>
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
TensorIPCSpec::TensorIPCSpec(
    torch::IntArrayRef tensor_size_param,
    torch::IntArrayRef tensor_stride_param,
    std::int64_t tensor_offset_param,
    torch::Dtype dtype_param,
    bool requires_grad_param,
    std::int32_t storage_device_param,
    std::string storage_handle_param,
    std::uint64_t storage_size_bytes_param,
    std::uint64_t storage_offset_bytes_param,
    std::string ref_counter_handle_param,
    std::uint64_t ref_counter_offset_param,
    cudaIpcEventHandle_t event_handle_param,
    bool event_sync_required_param)
    : tensor_size(tensor_size_param),
      tensor_stride(tensor_stride_param),
      tensor_offset(tensor_offset_param),
      dtype(dtype_param),
      requires_grad(requires_grad_param),
      storage_device(storage_device_param),
      storage_handle(std::move(storage_handle_param)),
      storage_size_bytes(storage_size_bytes_param),
      storage_offset_bytes(storage_offset_bytes_param),
      ref_counter_handle(std::move(ref_counter_handle_param)),
      ref_counter_offset(ref_counter_offset_param),
      event_handle(event_handle_param),
      event_sync_required(event_sync_required_param) {
}
//==============================================================================
// Implementation matches THPStorage_shareCuda, minus the Python-specific portions.
TensorIPCSpec PrepareTensorIPCSpec(const torch::Tensor &x) {
    auto& storage = x.storage();
    ASSERT_VALID_RUNTIME(storage.device_type() == at::kCUDA, "PrepareTensorIPCSpec: only available on CUDA");
    c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();
    ASSERT_VALID_RUNTIME(!storage_impl->received_cuda(), "Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending.");

    std::string storage_handle;
    uint64_t storage_size_bytes = storage.nbytes();
    uint64_t storage_offset_bytes = 0;
    std::string ref_counter_handle;
    uint64_t ref_counter_offset = 0;
    at::DeviceGuard device_guard(storage.device());
    cudaIpcEventHandle_t event_handle{};
    bool event_sync_required = false;
    if (storage.data()) {
        auto shandle = c10::cuda::CUDACachingAllocator::shareIpcHandle(storage.mutable_data());
        storage_handle = shandle.handle;
        storage_offset_bytes = shandle.offset;

        // Put Storage Data behind new ref counting context
        at::DataPtr sent_data_ptr = torch::GetNewRefCountedSentData(
            storage.mutable_data(), storage.device());
        auto old_data_ptr = storage.set_data_ptr(std::move(sent_data_ptr));
        auto sent_data =
        static_cast<torch::CudaIPCSentData*>(storage.data_ptr().get_context());
        sent_data->set_original_ptr(std::move(old_data_ptr));
        ref_counter_handle = sent_data->handle();
        ref_counter_offset = sent_data->offset();

        event_sync_required = sent_data->event_sync_required_;
        if (sent_data->event_sync_required_) {
            C10_CUDA_CHECK(
                cudaIpcGetEventHandle(&event_handle, sent_data->event_));
        }
    }
    
    return TensorIPCSpec(
        x.sizes(),
        x.strides(),
        x.storage_offset(),
        torch::typeMetaToScalarType(x.dtype()),
        x.requires_grad(),
        storage.device().index(),
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required
    );
}
} // namespace setu::commons::utils
//==============================================================================