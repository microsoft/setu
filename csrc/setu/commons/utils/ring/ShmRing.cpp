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
#include "commons/utils/ring/ShmRing.h"
//==============================================================================
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::commons::utils::ring {
//==============================================================================

void* ShmRing::CreateRaw(const std::string& shm_name, std::size_t total_size) {
  std::int32_t fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
  ASSERT_VALID_RUNTIME(fd >= 0, "shm_open(CREATE) failed for '{}': {}",
                       shm_name, strerror(errno));

  std::int32_t ret = ftruncate(fd, static_cast<off_t>(total_size));
  if (ret != 0) {
    close(fd);
    RAISE_RUNTIME_ERROR("ftruncate failed for '{}': {}", shm_name,
                        strerror(errno));
  }

  void* ptr =
      mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  ASSERT_VALID_RUNTIME(ptr != MAP_FAILED, "mmap failed for '{}': {}", shm_name,
                       strerror(errno));

  // Zero-initialize entire region before header setup
  std::memset(ptr, 0, total_size);

  LOG_DEBUG("ShmRing::Create: name={}, total_size={}", shm_name, total_size);
  return ptr;
}

void* ShmRing::OpenRaw(const std::string& shm_name, std::size_t total_size) {
  std::int32_t fd = shm_open(shm_name.c_str(), O_RDWR, 0600);
  ASSERT_VALID_RUNTIME(fd >= 0, "shm_open(OPEN) failed for '{}': {}", shm_name,
                       strerror(errno));

  void* ptr =
      mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  ASSERT_VALID_RUNTIME(ptr != MAP_FAILED, "mmap failed for '{}': {}", shm_name,
                       strerror(errno));

  LOG_DEBUG("ShmRing::Open: name={}, total_size={}", shm_name, total_size);
  return ptr;
}

void ShmRing::Destroy(const std::string& shm_name, void* ptr,
                      std::size_t size) {
  if (ptr != nullptr && ptr != MAP_FAILED) {
    munmap(ptr, size);
  }
  shm_unlink(shm_name.c_str());
  LOG_DEBUG("ShmRing::Destroy: name={}", shm_name);
}

std::string ShmRing::GenerateShmName(const std::string& prefix,
                                     const std::string& identity) {
  auto hash = std::hash<std::string>{}(identity);
  return std::format("/{}_{:016x}", prefix, hash);
}

std::uint32_t ShmRing::NextPowerOf2(std::uint32_t v) {
  if (v == 0) return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

//==============================================================================
}  // namespace setu::commons::utils::ring
//==============================================================================
