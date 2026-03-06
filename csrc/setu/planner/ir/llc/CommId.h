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
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::planner::ir::llc {
//==============================================================================
using setu::commons::utils::BinaryBuffer;
using setu::commons::utils::BinaryRange;
using setu::commons::utils::BinaryReader;
using setu::commons::utils::BinaryWriter;
//==============================================================================

/// Compile-time constraint for native communicator identifier types that can
/// be stored inside the type-erased CommId.
template <typename T>
concept NativeCommId = std::is_trivially_copyable_v<T> && (sizeof(T) <= 128);

/// Backend-agnostic communicator identifier.
///
/// Stores up to 128 bytes of opaque data — enough for any known backend
/// (e.g. ncclUniqueId is exactly 128 bytes).  Conversion to/from native
/// identifier types is concept-constrained via NativeCommId.
struct CommId {
  static constexpr std::size_t kMaxBytes = 128;

  std::array<char, kMaxBytes> data{};

  /// Construct a CommId from a backend-native identifier.
  template <NativeCommId T>
  [[nodiscard]] static CommId From(const T& native) {
    CommId id;
    std::memcpy(id.data.data(), &native, sizeof(T));
    return id;
  }

  /// Extract the backend-native identifier from a CommId.
  template <NativeCommId T>
  [[nodiscard]] T As() const {
    T native;
    std::memcpy(&native, data.data(), sizeof(T));
    return native;
  }

  [[nodiscard]] bool operator==(const CommId& other) const {
    return data == other.data;
  }

  [[nodiscard]] std::string ToString() const {
    std::string hex;
    for (std::size_t i = 0; i < 8; ++i) {
      hex += std::format("{:02x}", static_cast<unsigned char>(data[i]));
    }
    return std::format("CommId({}...)", hex);
  }

  void Serialize(BinaryBuffer& buffer) const {
    BinaryWriter writer(buffer);
    writer.Write(data);
  }

  [[nodiscard]] static CommId Deserialize(const BinaryRange& range) {
    BinaryReader reader(range);
    CommId id;
    id.data = reader.Read<std::array<char, kMaxBytes>>();
    return id;
  }
};

/// Hash support for CommId (e.g. for use as unordered_map key).
struct CommIdHash {
  std::size_t operator()(const CommId& id) const noexcept {
    // FNV-1a over the raw bytes
    std::size_t hash = 14695981039346656037ULL;
    for (auto c : id.data) {
      hash ^= static_cast<std::size_t>(static_cast<unsigned char>(c));
      hash *= 1099511628211ULL;
    }
    return hash;
  }
};

//==============================================================================
}  // namespace setu::planner::ir::llc
//==============================================================================
