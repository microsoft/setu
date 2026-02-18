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
#pragma once
//==============================================================================
#include "commons/StdCommon.h"
#include "commons/Types.h"
//==============================================================================
#include "planner/RegisterSet.h"
//==============================================================================
namespace setu::node_manager::worker {
//==============================================================================
using setu::commons::DevicePtr;
using setu::planner::RegisterSet;
//==============================================================================

/// Allocated register file: a RegisterSet spec paired with device pointers.
///
/// The worker creates a RegisterFile from a RegisterSet during Setup(),
/// allocates device memory for each register, and owns the pointers for
/// the lifetime of the file.  Call Allocate() to allocate and Free() to
/// release; the destructor calls Free() automatically.
class RegisterFile {
 public:
  RegisterFile() = default;

  explicit RegisterFile(RegisterSet spec) : spec_(std::move(spec)) {
    ptrs_.resize(spec_.NumRegisters(), nullptr);
  }

  ~RegisterFile() { Free(); }

  RegisterFile(const RegisterFile&) = delete;
  RegisterFile& operator=(const RegisterFile&) = delete;
  RegisterFile(RegisterFile&& other) noexcept
      : spec_(std::move(other.spec_)), ptrs_(std::move(other.ptrs_)) {
    other.ptrs_.clear();
  }
  RegisterFile& operator=(RegisterFile&& other) noexcept {
    if (this != &other) {
      Free();
      spec_ = std::move(other.spec_);
      ptrs_ = std::move(other.ptrs_);
      other.ptrs_.clear();
    }
    return *this;
  }

  /// Allocate device memory for all registers.  Must be called on the
  /// correct CUDA device (after cudaSetDevice).  Implemented in the .cpp
  /// file to keep CUDA includes out of this header.
  void Allocate();

  /// Free all allocated device memory.
  void Free();

  /// Device pointer for the register at the given index.
  [[nodiscard]] DevicePtr GetPtr(std::uint32_t index) const {
    return ptrs_.at(index);
  }

  /// Size in bytes of the register at the given index.
  [[nodiscard]] std::size_t SizeBytes(std::uint32_t index) const {
    return spec_.SizeBytes(index);
  }

  /// Number of registers.
  [[nodiscard]] std::uint32_t NumRegisters() const {
    return spec_.NumRegisters();
  }

  /// Whether any registers are defined.
  [[nodiscard]] bool Empty() const { return spec_.Empty(); }

  /// Access the underlying spec.
  [[nodiscard]] const RegisterSet& Spec() const { return spec_; }

 private:
  RegisterSet spec_;
  std::vector<DevicePtr> ptrs_;
};

//==============================================================================
}  // namespace setu::node_manager::worker
//==============================================================================
