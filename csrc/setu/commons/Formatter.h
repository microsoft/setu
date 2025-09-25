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
/**
 * @file Formatter.h
 * @brief Custom formatters for std::format support
 *
 * This file contains std::formatter specializations for various types used
 * throughout the Setu codebase, enabling them to be used with std::format
 * and related formatting functions.
 *
 * @defgroup Formatters Custom Formatter Specializations
 * @{
 */
//==============================================================================
#include "commons/BoostCommon.h"
#include "commons/Constants.h"
#include "commons/StdCommon.h"
#include "commons/TorchCommon.h"
//==============================================================================
namespace setu::commons {
//==============================================================================
[[nodiscard]] inline constexpr std::string_view ExtractTypeName(
    std::string_view pretty) {
  constexpr std::string_view k = "T = ";
  const std::size_t start = pretty.find(k) + k.size();
  const std::size_t end = pretty.rfind(']');
  return pretty.substr(start, end - start);
}
//==============================================================================
template <typename T>
[[nodiscard]] constexpr std::string_view GetTypeName() {
  return ExtractTypeName(__PRETTY_FUNCTION__);
}
//==============================================================================
template <typename T>
concept Printable = requires(const T& t) {
  { t.ToString() } -> std::convertible_to<std::string>;
};  // NOLINT(readability/braces)
//==============================================================================
//  Type traits for formatter template dispatch
//==============================================================================
template <typename U>
struct IsVector : std::false_type {};
template <typename U>
struct IsVector<std::vector<U>> : std::true_type {};
//==============================================================================
template <typename U>
struct IsOptional : std::false_type {};
template <typename U>
struct IsOptional<std::optional<U>> : std::true_type {};
//==============================================================================
template <typename U>
struct IsSharedPtr : std::false_type {};
template <typename U>
struct IsSharedPtr<std::shared_ptr<U>> : std::true_type {};
//==============================================================================
template <typename U>
struct IsUniquePtr : std::false_type {};
template <typename U>
struct IsUniquePtr<std::unique_ptr<U>> : std::true_type {};
//==============================================================================
template <typename U>
struct IsMap : std::false_type {};
template <typename K, typename V>
struct IsMap<std::unordered_map<K, V>> : std::true_type {};
template <typename K, typename V>
struct IsMap<std::map<K, V>> : std::true_type {};
//==============================================================================
template <typename U>
struct IsSet : std::false_type {};
template <typename U>
struct IsSet<std::unordered_set<U>> : std::true_type {};
template <typename U>
struct IsSet<std::set<U>> : std::true_type {};
//==============================================================================
template <typename U>
struct IsPair : std::false_type {};
template <typename T1, typename T2>
struct IsPair<std::pair<T1, T2>> : std::true_type {};
//==============================================================================
template <typename U>
struct IsEnum : std::false_type {};
template <typename U>
  requires std::is_enum_v<U>
struct IsEnum<U> : std::true_type {};
//==============================================================================
// Concept for basic formattable types (primitives + string types)
//==============================================================================
template <typename T>
concept BasicFormattable =
    std::is_arithmetic_v<T> || std::is_same_v<T, std::string> ||
    std::is_same_v<T, std::string_view> || std::is_same_v<T, const char*> ||
    std::is_enum_v<T>;
//==============================================================================
// Generic format helper function using template dispatch
//==============================================================================
template <typename T>
[[nodiscard]] std::string FormatValue(const T& value) {
  // Template dispatch based on type characteristics
  // Order matters: check specific types before general ones

  if constexpr (std::is_same_v<T, std::nullptr_t>) {
    // nullptr handling
    return kNullString;

  } else if constexpr (BasicFormattable<T>) {
    // Basic types that std::format can handle directly
    return std::format("{}", value);

  } else if constexpr (Printable<T>) {
    // Types with ToString() method
    return value.ToString();

  } else if constexpr (IsOptional<T>::value) {
    // Optional types: show value or "null"
    return value.has_value() ? FormatValue(*value) : kNullString;

  } else if constexpr (IsVector<T>::value) {
    // Vector types: [elem1, elem2, ...]
    std::string result = "[";
    for (std::size_t i = 0; i < value.size(); ++i) {
      if (i > 0) result += ", ";
      result += FormatValue(value[i]);
    }
    result += "]";
    return result;

  } else if constexpr (IsSet<T>::value) {
    // Set types: {elem1, elem2, ...}
    std::string result = "{";
    std::size_t i = 0;
    for (const auto& elem : value) {
      if (i > 0) result += ", ";
      result += FormatValue(elem);
      ++i;
    }
    result += "}";
    return result;

  } else if constexpr (IsMap<T>::value) {
    // Map types: {key1: value1, key2: value2, ...}
    std::string result = "{";
    std::size_t i = 0;
    for (const auto& [key, val] : value) {
      if (i > 0) result += ", ";
      result += FormatValue(key) + ": " + FormatValue(val);
      ++i;
    }
    result += "}";
    return result;

  } else if constexpr (IsPair<T>::value) {
    // Pair types: (first, second)
    return std::format("({}, {})", FormatValue(value.first),
                       FormatValue(value.second));

  } else if constexpr (IsSharedPtr<T>::value) {
    // Shared pointer types: show pointee or "null"
    using PointeeType = typename T::element_type;
    if constexpr (Printable<PointeeType>) {
      return value ? value->ToString() : kNullString;
    } else {
      return value ? FormatValue(*value) : kNullString;
    }

  } else if constexpr (IsUniquePtr<T>::value) {
    // Unique pointer types: show pointee or "null"
    using PointeeType = typename T::element_type;
    if constexpr (Printable<PointeeType>) {
      return value ? value->ToString() : kNullString;
    } else {
      return value ? FormatValue(*value) : kNullString;
    }

  } else if constexpr (std::is_pointer_v<T>) {
    // Raw pointer types: show pointee or "null"
    using PointeeType = std::remove_pointer_t<T>;
    if constexpr (Printable<PointeeType>) {
      return value ? value->ToString() : kNullString;
    } else {
      return value ? FormatValue(*value) : kNullString;
    }

  } else {
    // Fallback: try to use std::format directly, or show type name
    if constexpr (requires { std::format("{}", value); }) {
      return std::format("{}", value);
    } else {
      return std::format("[{} object]", GetTypeName<T>());
    }
  }
}

//==============================================================================
}  // namespace setu::commons
//==============================================================================
namespace std {
//==============================================================================
// Formatter for c10::ArrayRef<T> (for printing tensor sizes)
//==============================================================================
/// @brief Formatter specialization for c10::ArrayRef
template <typename T>
struct formatter<c10::ArrayRef<T>, char> {
  // For simplicity, no custom parse options.
  // The parse method is called to parse format specifiers (e.g., alignment,
  // width) For this example, we don't support any custom format specifiers for
  // ArrayRef. It just needs to find the closing '}'.
  constexpr auto parse(std::format_parse_context& ctx) {
    auto it = ctx.begin();
    auto end = ctx.end();
    if (it != end && *it != '}') {
      // You could add parsing for custom options here if needed
      // For now, just advance to the closing brace or end
      while (it != end && *it != '}') {
        ++it;
      }
    }
    return it;
  }

  // The format method does the actual formatting
  template <typename FormatContext>
  auto format(const c10::ArrayRef<T>& arr_ref, FormatContext& ctx) const {
    auto out = ctx.out();
    out = std::format_to(out, "[");
    if (!arr_ref.empty()) {
      // This relies on T being formattable by std::format
      out = std::format_to(out, "{}", arr_ref[0]);
      for (size_t i = 1; i < arr_ref.size(); ++i) {
        out = std::format_to(out, ", {}", arr_ref[i]);
      }
    }
    out = std::format_to(out, "]");
    return out;
  }
};
//==============================================================================
// Formatter for c10::ScalarType (for printing dtype)
//==============================================================================
/// @brief Formatter specialization for PyTorch scalar types
/// @ingroup Formatters
template <>
struct formatter<c10::ScalarType, char> {
  // No custom parse options for dtype
  constexpr auto parse(std::format_parse_context& ctx) {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && *it != '}') {
      // If you wanted to allow specifiers, parse them here.
      // For now, just expect it to be empty or find '}'.
      while (it != end && *it != '}') ++it;
    }
    return it;
  }

  template <typename FormatContext>
  auto format(const c10::ScalarType& dtype, FormatContext& ctx) const {
    // c10::toString(dtype) returns a const char* like "Float" or "Long"
    // To get "torch.float32", we might need a small mapping or use it as is.
    // For now, let's prepend "torch." to the output of c10::toString
    // Note: c10::toString gives "Float", "Double", "Long", etc.
    // PyTorch Python API shows "torch.float32", "torch.int64"
    // We can create a small helper or live with the C++ style names.

    // Let's try to map to Python-like names:
    std::string_view s;
    switch (dtype) {
      case c10::ScalarType::Byte:
        s = "torch.uint8";
        break;
      case c10::ScalarType::Char:
        s = "torch.int8";
        break;
      case c10::ScalarType::Short:
        s = "torch.int16";
        break;
      case c10::ScalarType::Int:
        s = "torch.int32";
        break;
      case c10::ScalarType::Long:
        s = "torch.int64";
        break;
      case c10::ScalarType::Half:
        s = "torch.float16";
        break;
      case c10::ScalarType::Float:
        s = "torch.float32";
        break;
      case c10::ScalarType::Double:
        s = "torch.float64";
        break;
      case c10::ScalarType::ComplexHalf:
        s = "torch.complex32";
        break;
      case c10::ScalarType::ComplexFloat:
        s = "torch.complex64";
        break;
      case c10::ScalarType::ComplexDouble:
        s = "torch.complex128";
        break;
      case c10::ScalarType::Bool:
        s = "torch.bool";
        break;
      case c10::ScalarType::QInt8:
        s = "torch.qint8";
        break;
      case c10::ScalarType::QUInt8:
        s = "torch.quint8";
        break;
      case c10::ScalarType::QInt32:
        s = "torch.qint32";
        break;
      case c10::ScalarType::BFloat16:
        s = "torch.bfloat16";
        break;
      // c10::ScalarType::QUInt4x2, c10::ScalarType::QUInt2x4 might not have
      // direct torch. counterparts or are newer
      default:
        s = c10::toString(dtype);  // Fallback
    }
    return std::format_to(ctx.out(), "{}", s);
  }
};
//==============================================================================
// Formatter for at::Tensor (for printing tensor values)
//==============================================================================
/// @brief Formatter specialization for PyTorch tensors
/// @ingroup Formatters
template <>
struct formatter<torch::Tensor, char> : formatter<std::string_view, char> {
 public:
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return formatter<std::string_view, char>::parse(ctx);
  }

  template <typename FormatContext>
  auto format(const torch::Tensor& tensor, FormatContext& ctx) const {
    std::string tensor_representation;

    std::ostringstream oss;
    if (!tensor.defined()) {
      oss << "[ Tensor (undefined) ]";
    } else {
      oss << tensor;
    }
    tensor_representation = oss.str();

    return formatter<std::string_view, char>::format(tensor_representation,
                                                     ctx);
  }
};
//==============================================================================
// Formatter for boost::concurrent::sync_queue<T>
// (Handles setu::Queue<T>)
//==============================================================================
/// @brief Formatter specialization for Boost concurrent queue
/// @ingroup Formatters
template <typename T>
struct formatter<boost::concurrent::sync_queue<T>, char> {
  // No custom parsing options for sync_queue
  constexpr auto parse(std::format_parse_context& ctx) {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && *it != '}') {
      // Consume any characters until '}' or end, if any are present.
      // This basic formatter doesn't interpret them.
      while (it != end && *it != '}') {
        ++it;
      }
    }
    return it;
  }

  template <typename FormatContext>
  auto format(const boost::concurrent::sync_queue<T>& q,
              FormatContext& ctx) const {
    // Note: Getting a user-friendly string for type T is non-trivial without
    // RTTI/external libs. The "T" in "Queue<T>" is symbolic here.
    return std::format_to(ctx.out(), "setu::Queue<T>(status: {}, size: {})",
                          q.closed() ? "closed" : "open",
                          q.size());  // .size() and .closed() are const
  }
};
//==============================================================================
// Formatter for boost::concurrent::sync_priority_queue<Value, Container,
// Compare> (Handles setu::PriorityQueue<...>)
//==============================================================================
/// @brief Formatter specialization for Boost concurrent priority queue
/// @ingroup Formatters
template <typename Value, typename Container, typename Compare>
struct formatter<
    boost::concurrent::sync_priority_queue<Value, Container, Compare>, char> {
  // No custom parsing options for sync_priority_queue
  constexpr auto parse(std::format_parse_context& ctx) {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && *it != '}') {
      while (it != end && *it != '}') {
        ++it;
      }
    }
    return it;
  }

  template <typename FormatContext>
  auto format(const boost::concurrent::sync_priority_queue<Value, Container,
                                                           Compare>& q,
              FormatContext& ctx) const {
    // Similar to sync_queue, "Value,..." is symbolic.
    return std::format_to(
        ctx.out(), "setu::PriorityQueue<Value,...>(status: {}, size: {})",
        q.closed() ? "closed" : "open",
        q.size());  // .size() and .closed() are const
  }
};
//==============================================================================
// Formatter for Boost::dynamic_bitset<>
//==============================================================================
/// @brief Formatter specialization for Boost::dynamic_bitset
/// @ingroup Formatters
template <>
struct formatter<boost::dynamic_bitset<>, char> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const boost::dynamic_bitset<>& bitset, FormatContext& ctx) const {
    std::basic_string<char, std::char_traits<char>> s;
    boost::to_string(bitset, s);
    return std::format_to(ctx.out(), "{}", s);
  }
};
//==============================================================================
// Formatter for Printable types (types with ToString() method)
//==============================================================================
/// @brief Formatter specialization for types with ToString() method
/// @ingroup Formatters
template <typename T>
  requires setu::commons::Printable<T>
struct formatter<T> : formatter<string> {
  auto format(const T& obj, format_context& ctx) const {
    return formatter<string>::format(obj.ToString(), ctx);
  }
};
//==============================================================================
// Generic formatter for all types using our FormatValue function
//==============================================================================
/// @brief Generic formatter specialization for custom types
/// @ingroup Formatters
template <typename T>
  requires(!setu::commons::BasicFormattable<T> &&
           !setu::commons::Printable<T> &&
           !std::is_same_v<T, c10::ScalarType> &&
           !std::is_same_v<T, torch::Tensor> &&
           !std::is_same_v<T, boost::dynamic_bitset<>> &&
           !std::is_same_v<std::remove_cv_t<T>,
                           boost::concurrent::sync_queue<
                               typename std::remove_cv_t<T>::value_type>> &&
           !std::is_same_v<std::remove_cv_t<T>,
                           boost::concurrent::sync_priority_queue<
                               typename std::remove_cv_t<T>::value_type,
                               typename std::remove_cv_t<T>::container_type,
                               typename std::remove_cv_t<T>::value_compare>>)
struct formatter<T> : formatter<string> {
  auto format(const T& obj, format_context& ctx) const {
    return formatter<string>::format(setu::commons::FormatValue(obj), ctx);
  }
};
//==============================================================================
// Specific formatter specializations for container types to ensure they use our
// FormatValue
//==============================================================================
/// @brief Formatter specialization for std::vector
/// @ingroup Formatters
template <typename T>
struct formatter<std::vector<T>> : formatter<string> {
  auto format(const std::vector<T>& vec, format_context& ctx) const {
    return formatter<string>::format(setu::commons::FormatValue(vec), ctx);
  }
};
//==============================================================================
/// @brief Formatter specialization for std::optional
/// @ingroup Formatters
template <typename T>
struct formatter<std::optional<T>> : formatter<string> {
  auto format(const std::optional<T>& opt, format_context& ctx) const {
    return formatter<string>::format(setu::commons::FormatValue(opt), ctx);
  }
};
//==============================================================================
/// @brief Formatter specialization for std::unordered_map
/// @ingroup Formatters
template <typename K, typename V>
struct formatter<std::unordered_map<K, V>> : formatter<string> {
  auto format(const std::unordered_map<K, V>& map, format_context& ctx) const {
    return formatter<string>::format(setu::commons::FormatValue(map), ctx);
  }
};
//==============================================================================
/// @brief Formatter specialization for std::map
/// @ingroup Formatters
template <typename K, typename V>
struct formatter<std::map<K, V>> : formatter<string> {
  auto format(const std::map<K, V>& map, format_context& ctx) const {
    return formatter<string>::format(setu::commons::FormatValue(map), ctx);
  }
};
//==============================================================================
/// @brief Formatter specialization for std::set
/// @ingroup Formatters
template <typename T>
struct formatter<std::set<T>> : formatter<string> {
  auto format(const std::set<T>& set, format_context& ctx) const {
    return formatter<string>::format(setu::commons::FormatValue(set), ctx);
  }
};
//==============================================================================
/// @brief Formatter specialization for std::unordered_set
/// @ingroup Formatters
template <typename T>
struct formatter<std::unordered_set<T>> : formatter<string> {
  auto format(const std::unordered_set<T>& set, format_context& ctx) const {
    return formatter<string>::format(setu::commons::FormatValue(set), ctx);
  }
};
//==============================================================================
/// @brief Formatter specialization for std::pair
/// @ingroup Formatters
template <typename T1, typename T2>
struct formatter<std::pair<T1, T2>> : formatter<string> {
  auto format(const std::pair<T1, T2>& pair, format_context& ctx) const {
    return formatter<string>::format(setu::commons::FormatValue(pair), ctx);
  }
};
//==============================================================================
// formatter specialization for shared_ptr
/// @brief Formatter specialization for std::shared_ptr
/// @ingroup Formatters
template <typename T>
struct formatter<shared_ptr<T>> : formatter<string> {
  auto format(const shared_ptr<T>& ptr, format_context& ctx) const {
    return formatter<string>::format(setu::commons::FormatValue(ptr), ctx);
  }
};
//==============================================================================
// formatter specialization for unique_ptr
/// @brief Formatter specialization for std::unique_ptr
/// @ingroup Formatters
template <typename T>
struct formatter<unique_ptr<T>> : formatter<string> {
  auto format(const unique_ptr<T>& ptr, format_context& ctx) const {
    return formatter<string>::format(setu::commons::FormatValue(ptr), ctx);
  }
};
//==============================================================================
}  // namespace std
//==============================================================================
