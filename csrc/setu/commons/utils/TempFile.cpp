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
#include "commons/utils/TempFile.h"
//==============================================================================
#include "commons/Logging.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================

TempFile::TempFile(const std::string& base_path, const std::string& filename,
                   int flags, mode_t mode)
    : fd_(-1) {
  // Create base directory if it doesn't exist
  std::filesystem::path base_dir(base_path);
  if (!base_dir.empty() && !std::filesystem::exists(base_dir)) {
    std::filesystem::create_directories(base_dir);
  }

  // Generate unique filename with timestamp and random suffix
  auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  auto random_suffix = std::random_device{}();

  std::filesystem::path temp_path =
      base_dir /
      std::format("{}.{}.{}.tmp", filename, timestamp, random_suffix);

  filepath_ = temp_path.string();

  // Open the file with O_EXCL to ensure it's newly created
  fd_ = ::open(filepath_.c_str(), flags, mode);
  if (fd_ < 0) {
    RAISE_RUNTIME_ERROR("Failed to create temporary file '{}': {}", filepath_,
                        std::strerror(errno));
  }

  LOG_DEBUG("Created temporary file: {}", filepath_);
}

//==============================================================================

TempFile::TempFile(TempFile&& other) noexcept
    : fd_(other.fd_), filepath_(std::move(other.filepath_)) {
  other.fd_ = -1;  // Prevent the moved-from object from closing/unlinking
  other.filepath_.clear();
}

//==============================================================================

TempFile& TempFile::operator=(TempFile&& other) noexcept {
  if (this != &other) {
    CleanUp();  // Close and unlink current file if open
    fd_ = other.fd_;
    filepath_ = std::move(other.filepath_);
    other.fd_ = -1;
    other.filepath_.clear();
  }
  return *this;
}

//==============================================================================

TempFile::~TempFile() { CleanUp(); }

//==============================================================================

std::int32_t TempFile::GetFileDescriptor() const { return fd_; }

//==============================================================================

const std::string& TempFile::GetFilePath() const { return filepath_; }

//==============================================================================

std::size_t TempFile::GetFileSize() const {
  if (fd_ < 0 || filepath_.empty()) {
    return 0;  // File not open
  }

  std::error_code ec;
  auto size = std::filesystem::file_size(filepath_, ec);

  if (ec) {
    LOG_WARNING("Failed to get file size for '{}': {}", filepath_,
                ec.message());
    return 0;  // Error occurred
  }

  return static_cast<std::size_t>(size);
}

//==============================================================================

void TempFile::CleanUp() {
  if (fd_ >= 0) {
    if (::close(fd_) < 0) {
      LOG_WARNING("Failed to close file descriptor {} for '{}': {}", fd_,
                  filepath_, std::strerror(errno));
    }
    fd_ = -1;

    if (!filepath_.empty()) {
      std::error_code ec;
      std::filesystem::remove(filepath_, ec);
      if (ec) {
        LOG_WARNING("Failed to remove temporary file '{}': {}", filepath_,
                    ec.message());
      } else {
        LOG_DEBUG("Removed temporary file: {}", filepath_);
      }
      filepath_.clear();
    }
  }
}

//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
