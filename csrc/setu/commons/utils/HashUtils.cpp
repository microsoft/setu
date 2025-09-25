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
#include "commons/utils/HashUtils.h"
//==============================================================================
#include "commons/BoostCommon.h"
#include "commons/Logging.h"
#include "commons/StdCommon.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
[[nodiscard]] std::string GetHash(const std::string& data,
                                  std::size_t hash_length) {
  ASSERT_VALID_RUNTIME(hash_length > 0 && hash_length <= 32,
                       "Invalid hash length");

  boost::uuids::detail::md5 md5;
  md5.process_bytes(data.data(), data.size());

  // Boost MD5 expects unsigned int[4] instead of unsigned char[16]
  unsigned char digest[16];
  md5.get_digest(digest);

  // Convert the digest to a string
  std::stringstream ss;
  for (int i = 0; i < 4; ++i) {
    ss << std::hex << std::setfill('0') << std::setw(8) << digest[i];
  }

  // Return the substring of specified length
  return ss.str().substr(0, hash_length);
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
