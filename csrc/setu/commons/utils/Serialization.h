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
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/Types.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::BinaryBuffer;
using setu::commons::BinaryIterator;
using setu::commons::BinaryRange;
using setu::commons::GetTypeName;
//==============================================================================
//  Concepts
//==============================================================================
//  A Serializable type provides const Serialize(...) and static
//  Deserialize(...) We accept const‑qualified template parameters
//  transparently.
template <typename T>
concept Serializable = requires(const std::remove_const_t<T>& obj,
                                BinaryBuffer& buf, BinaryRange range) {
  { obj.Serialize(buf) } -> std::same_as<void>;
  {
    std::remove_const_t<T>::Deserialize(range)
  } -> std::same_as<std::remove_const_t<T>>;
};  // NOLINT(readability/braces)
//==============================================================================
template <typename T>
concept TriviallySerializable =
    std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> &&
    !std::is_pointer_v<T>;
//==============================================================================
//  Type traits for template dispatch
//==============================================================================
template <typename U>
struct IsVector : std::false_type {};
template <typename U>
struct IsVector<std::vector<U>> : std::true_type {};

template <typename U>
struct IsDeque : std::false_type {};
template <typename U>
struct IsDeque<std::deque<U>> : std::true_type {};

template <typename U>
struct IsOptional : std::false_type {};
template <typename U>
struct IsOptional<std::optional<U>> : std::true_type {};

template <typename U>
struct IsSharedPtr : std::false_type {};
template <typename U>
struct IsSharedPtr<std::shared_ptr<U>> : std::true_type {};

template <typename U>
struct IsMap : std::false_type {};
template <typename K, typename V>
struct IsMap<std::unordered_map<K, V>> : std::true_type {};

template <typename U>
struct IsEnum : std::false_type {};
template <typename U>
  requires std::is_enum_v<U>
struct IsEnum<U> : std::true_type {};

template <typename U>
struct IsVariant : std::false_type {};
template <typename... Types>
struct IsVariant<std::variant<Types...>> : std::true_type {};

//==============================================================================
//  BinaryWriter – append‑only
//==============================================================================
class BinaryWriter {
 public:
  [[nodiscard]] explicit BinaryWriter(BinaryBuffer& buffer)
      : buffer_(buffer), init_size_(buffer.size()) {}

  template <typename T>
  void Write(const T& value) {
    // Template dispatch based on type characteristics at compile time
    // Order matters: check specific types before general ones

    if constexpr (std::is_same_v<T, std::string> ||
                  std::is_same_v<T, std::string_view>) {
      // String types: write size prefix + raw data
      Write<std::uint32_t>(static_cast<std::uint32_t>(value.size()));
      WriteBytes(reinterpret_cast<const std::uint8_t*>(value.data()),
                 value.size());

    } else if constexpr (IsEnum<T>::value) {
      // Enum types: convert to underlying type for serialization
      using UnderlyingType = std::underlying_type_t<T>;
      Write<UnderlyingType>(static_cast<UnderlyingType>(value));

    } else if constexpr (IsOptional<T>::value) {
      // Optional types: write presence flag + value if present
      Write<bool>(value.has_value());
      if (value) Write(*value);

    } else if constexpr (IsVector<T>::value) {
      // Vector types: write size + elements (optimized for POD types)
      using E = typename T::value_type;
      Write<std::uint32_t>(static_cast<std::uint32_t>(value.size()));
      if (value.empty()) return;
      if constexpr (TriviallySerializable<E>) {
        // POD types: bulk copy for performance
        WriteBytes(reinterpret_cast<const std::uint8_t*>(value.data()),
                   value.size() * sizeof(E));
      } else {
        // Complex types: serialize individually
        for (const auto& item : value) Write(item);
      }

    } else if constexpr (IsDeque<T>::value) {
      // Deque types: write size + elements (serialize individually as deques
      // don't guarantee contiguous storage)
      Write<std::uint32_t>(static_cast<std::uint32_t>(value.size()));
      if (value.empty()) return;
      // Always serialize individually for deques (no contiguous storage
      // guarantee)
      for (const auto& item : value) Write(item);

    } else if constexpr (IsSharedPtr<T>::value) {
      // Shared pointer types: write nullness flag + pointee if non-null
      Write<bool>(static_cast<bool>(value));
      if (value) Write(*value);

    } else if constexpr (IsVariant<T>::value) {
      // Variant types: write index + active alternative value
      Write<std::uint32_t>(static_cast<std::uint32_t>(value.index()));
      std::visit([this](const auto& alternative) { Write(alternative); },
                 value);

    } else if constexpr (IsMap<T>::value) {
      // Map types: write size + key-value pairs
      Write<std::uint32_t>(static_cast<std::uint32_t>(value.size()));
      for (const auto& [k, v] : value) {
        Write(k);
        Write(v);
      }

    } else if constexpr (Serializable<T>) {
      // Custom serializable types: delegate to object's method with size prefix
      BinaryBuffer tmp;
      value.Serialize(tmp);
      Write<std::uint32_t>(static_cast<std::uint32_t>(tmp.size()));
      WriteBytes(tmp.data(), tmp.size());

    } else if constexpr (TriviallySerializable<T>) {
      // POD types: direct memory copy
      const auto* data = reinterpret_cast<const std::uint8_t*>(&value);
      WriteBytes(data, sizeof(T));

    } else {
      // Compile-time error for unsupported types
      static_assert(sizeof(T) == 0,
                    "Unsupported type in BinaryWriter::Write()");
    }
  }

  template <typename... Args>
  void WriteFields(const Args&... args) {
    (Write(args), ...);
  }

  [[nodiscard]] std::size_t BytesWritten() const {
    return buffer_.size() - init_size_;
  }
  [[nodiscard]] BinaryBuffer& Buffer() { return buffer_; }

 private:
  void WriteBytes(const std::uint8_t* data, std::size_t size) {
    ASSERT_VALID_RUNTIME(size == 0 || data != nullptr,
                         "WriteBytes: null data with non‑zero size");
    buffer_.insert(buffer_.end(), data, data + size);
  }
  BinaryBuffer& buffer_;
  std::size_t init_size_;
};
//==============================================================================
//  BinaryReader – forward‑only
//==============================================================================
class BinaryReader {
 public:
  BinaryReader(BinaryIterator begin, BinaryIterator end)
      : cur_(begin), end_(end) {}
  explicit BinaryReader(const BinaryRange& range)
      : cur_(range.first), end_(range.second) {}

  template <typename T>
  [[nodiscard]] T Read() {
    // Template dispatch to mirror Write() logic exactly
    // Order must match Write() for correct deserialization

    if constexpr (std::is_const_v<T>) {
      // Handle const-qualified types by removing const
      return Read<std::remove_const_t<T>>();

    } else if constexpr (std::is_same_v<T, std::string>) {
      // String types: read size prefix + raw data
      const auto size = Read<std::uint32_t>();
      ASSERT_VALID_RUNTIME(cur_ + size <= end_,
                           "String overflow while reading {}",
                           GetTypeName<T>());
      std::string s(cur_, cur_ + size);
      cur_ += size;
      return s;

    } else if constexpr (IsEnum<T>::value) {
      // Enum types: read underlying type and cast back to enum
      using UnderlyingType = std::underlying_type_t<T>;
      return static_cast<T>(Read<UnderlyingType>());

    } else if constexpr (IsOptional<T>::value) {
      // Optional types: read presence flag + value if present
      using V = typename T::value_type;
      return Read<bool>() ? std::optional<V>(Read<V>()) : std::nullopt;

    } else if constexpr (IsVector<T>::value) {
      // Vector types: read size + elements (optimized for POD types)
      using E = typename T::value_type;
      const auto sz = Read<std::uint32_t>();
      T vec;
      vec.reserve(sz);
      if constexpr (TriviallySerializable<E>) {
        // POD types: bulk copy for performance
        const std::size_t bytes = static_cast<std::size_t>(sz) * sizeof(E);
        ASSERT_VALID_RUNTIME(cur_ + bytes <= end_, "Vector overflow reading {}",
                             GetTypeName<E>());
        vec.resize(sz);
        std::memcpy(vec.data(), &(*cur_), bytes);
        cur_ += bytes;
      } else {
        // Complex types: deserialize individually
        for (std::uint32_t i = 0; i < sz; ++i) vec.push_back(Read<E>());
      }
      return vec;

    } else if constexpr (IsDeque<T>::value) {
      // Deque types: read size + elements (deserialize individually as deques
      // don't guarantee contiguous storage)
      using E = typename T::value_type;
      const auto sz = Read<std::uint32_t>();
      T deq;
      // Always deserialize individually for deques (no contiguous storage
      // guarantee)
      for (std::uint32_t i = 0; i < sz; ++i) deq.push_back(Read<E>());
      return deq;

    } else if constexpr (IsSharedPtr<T>::value) {
      // Shared pointer types: read nullness flag + pointee if non-null
      using E = typename T::element_type;
      using B = std::remove_const_t<E>;
      return Read<bool>()
                 ? std::static_pointer_cast<E>(std::make_shared<B>(Read<B>()))
                 : nullptr;

    } else if constexpr (IsVariant<T>::value) {
      // Variant types: read index + construct appropriate alternative
      const auto index = Read<std::uint32_t>();
      return ReadVariantAtIndex<T>(index);

    } else if constexpr (IsMap<T>::value) {
      // Map types: read size + key-value pairs (ensure correct order!)
      using K = typename T::key_type;
      using V = typename T::mapped_type;
      const auto sz = Read<std::uint32_t>();
      T m;
      m.reserve(sz);
      for (std::uint32_t i = 0; i < sz; ++i) {
        // Read key and value in separate variables to avoid evaluation order
        // issues
        auto key = Read<K>();
        auto value = Read<V>();
        m.emplace(key, value);
      }
      return m;

    } else if constexpr (Serializable<T>) {
      // Custom serializable types: read size prefix + delegate to object's
      // method
      const auto bytes = Read<std::uint32_t>();
      ASSERT_VALID_RUNTIME(cur_ + bytes <= end_,
                           "Serializable overflow reading {}",
                           GetTypeName<T>());
      T obj =
          std::remove_const_t<T>::Deserialize(BinaryRange{cur_, cur_ + bytes});
      cur_ += bytes;
      return obj;

    } else if constexpr (TriviallySerializable<T>) {
      // POD types: direct memory copy
      ASSERT_VALID_RUNTIME(cur_ + sizeof(T) <= end_, "POD overflow reading {}",
                           GetTypeName<T>());
      T val;
      std::memcpy(&val, &(*cur_), sizeof(T));
      cur_ += sizeof(T);
      return val;

    } else {
      // Compile-time error for unsupported types
      static_assert(sizeof(T) == 0, "Unsupported type in BinaryReader::Read()");
    }
  }

  template <typename... Args>
  std::tuple<Args...> ReadFields() {
    return {Read<Args>()...};
  }

 private:
  template <typename VariantType>
  VariantType ReadVariantAtIndex(std::uint32_t index) {
    return ReadVariantAtIndexHelper<VariantType, 0>(index);
  }

  template <typename VariantType, std::size_t Index>
  VariantType ReadVariantAtIndexHelper(std::uint32_t index) {
    if constexpr (Index >= std::variant_size_v<VariantType>) {
      RAISE_RUNTIME_ERROR("Invalid variant index {} for type {}", index,
                          GetTypeName<VariantType>());
    } else {
      if (index == Index) {
        using AlternativeType = std::variant_alternative_t<Index, VariantType>;
        return VariantType(std::in_place_index<Index>, Read<AlternativeType>());
      } else {
        return ReadVariantAtIndexHelper<VariantType, Index + 1>(index);
      }
    }
  }

  BinaryIterator cur_, end_;
};
//==============================================================================
//  Free helpers
//==============================================================================
template <Serializable T>
[[nodiscard]] BinaryBuffer Serialize(const T& obj) {
  BinaryBuffer buf;
  obj.Serialize(buf);
  return buf;
}

template <Serializable T>
[[nodiscard]] std::remove_const_t<T> Deserialize(const BinaryBuffer& buf) {
  return std::remove_const_t<T>::Deserialize(
      BinaryRange{buf.begin(), buf.end()});
}

template <Serializable T>
[[nodiscard]] std::remove_const_t<T> Deserialize(
    std::span<const std::uint8_t> bytes) {
  BinaryBuffer tmp(bytes.begin(), bytes.end());
  return std::remove_const_t<T>::Deserialize(
      BinaryRange{tmp.begin(), tmp.end()});
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
