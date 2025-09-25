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
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
// Forward declaration of TempFile class
class TempFile;
using TempFilePtr = std::shared_ptr<TempFile>;
//==============================================================================

/**
 * @brief RAII wrapper for temporary files.
 *
 * This class handles the creation, opening, and automatic cleanup of temporary
 * files. The file is automatically closed and deleted when the TempFile object
 * is destroyed.
 */
class TempFile final {
 public:
  /**
   * @brief Creates a temporary file with the given base path and options.
   *
   * @param base_path Directory path where the temporary file will be created
   * @param filename Base name for the temporary file (timestamp and random
   *                 suffix will be appended)
   * @param flags File open flags (e.g., O_RDWR | O_CREAT | O_DIRECT)
   * @param mode File permissions (e.g., 0600)
   */
  TempFile(const std::string& base_path, const std::string& filename,
           int flags = O_RDWR | O_CREAT | O_EXCL, mode_t mode = 0600);

  /**
   * @brief Move constructor.
   */
  TempFile(TempFile&& other) noexcept;

  /**
   * @brief Move assignment operator.
   */
  TempFile& operator=(TempFile&& other) noexcept;

  /**
   * @brief Destructor. Automatically closes and deletes the file.
   */
  ~TempFile();

  // Deleted copy operations - temp files should not be copied
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  /**
   * @brief Get the file descriptor.
   * @return The file descriptor, or -1 if the file is not open
   */
  [[nodiscard]] std::int32_t GetFileDescriptor() const;

  /**
   * @brief Get the file path.
   * @return The absolute path to the temporary file
   */
  [[nodiscard]] const std::string& GetFilePath() const;

  /**
   * @brief Get the file size in bytes.
   * @return Size of the file in bytes, or 0 if an error occurs
   */
  [[nodiscard]] std::size_t GetFileSize() const;

 private:
  /**
   * @brief Close and unlink the temporary file.
   */
  void CleanUp();

  std::int32_t fd_;       // File descriptor
  std::string filepath_;  // Path to the temporary file
};

//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
