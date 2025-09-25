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
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
//  Concepts
//==============================================================================
/// A StructSerializable type exposes a const `JsonSerialize(JsonWriter&)`
/// method.
template <typename T>
concept StructSerializable =
    requires(const std::remove_const_t<T>& obj, class JsonWriter& writer) {
      { obj.JsonSerialize(writer) } -> std::same_as<void>;
    };  // NOLINT
//==============================================================================
//  Type traits – collision‑safe prefix (Json*)
//==============================================================================
template <typename T>
struct JsonIsVector : std::false_type {};
template <typename T>
struct JsonIsOptional : std::false_type {};
template <typename T>
struct JsonIsVariant : std::false_type {};
template <typename T>
struct JsonIsMap : std::false_type {};

template <typename U>
struct JsonIsVector<std::vector<U>> : std::true_type {};

template <typename U>
struct JsonIsOptional<std::optional<U>> : std::true_type {};

template <typename... Alts>
struct JsonIsVariant<std::variant<Alts...>> : std::true_type {};

template <typename K, typename V>
struct JsonIsMap<std::unordered_map<K, V>> : std::true_type {};
//==============================================================================
//  Helper utils
//==============================================================================
namespace detail {
inline std::string SafeDoubleToJson(double v) {
  if (!std::isfinite(v)) {
    LOG_WARNING("Replacing non‑finite double value {} with null in JSON", v);
    return "null";
  }
  return std::to_string(v);
}

inline std::string Escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += std::format("\\u{:04x}", static_cast<unsigned char>(c));
        } else {
          out += c;
        }
    }
  }
  return out;
}
}  // namespace detail
//==============================================================================
//  JsonWriter – Append‑only, mirrors BinaryWriter style.
//==============================================================================
class JsonWriter {
 public:
  explicit JsonWriter(std::string& buf) : buf_(buf) {}

  //-----------------------------------------------------------------------
  // Generic entry – SFINAE dispatch on `T` characteristics.
  //-----------------------------------------------------------------------
  template <typename T>
  void Write(const T& value) {
    if constexpr (std::is_same_v<T, std::string> ||
                  std::is_same_v<T, std::string_view>) {
      buf_ += '"' + detail::Escape(std::string(value)) + '"';

    } else if constexpr (std::is_same_v<T, const char*>) {
      buf_ += '"' + detail::Escape(value ? std::string(value) : std::string()) +
              '"';

    } else if constexpr (std::is_same_v<T, bool>) {
      buf_ += value ? "true" : "false";

    } else if constexpr (std::is_floating_point_v<T>) {
      buf_ += detail::SafeDoubleToJson(static_cast<double>(value));

    } else if constexpr (std::is_integral_v<T>) {
      buf_ += std::to_string(value);

    } else if constexpr (std::is_enum_v<T>) {
      using U = std::underlying_type_t<T>;
      buf_ += std::to_string(static_cast<U>(value));

    } else if constexpr (JsonIsOptional<T>::value) {
      if (value) {
        Write(*value);
      } else {
        buf_ += "null";
      }

    } else if constexpr (JsonIsVector<T>::value) {
      buf_ += '[';
      for (std::size_t i = 0; i < value.size(); ++i) {
        if (i) buf_ += ',';
        Write(value[i]);
      }
      buf_ += ']';

    } else if constexpr (JsonIsVariant<T>::value) {
      std::visit([this](const auto& alt) { Write(alt); }, value);

    } else if constexpr (JsonIsMap<T>::value) {
      buf_ += '{';
      bool first = true;
      for (const auto& [k, v] : value) {
        if (!first) buf_ += ',';
        first = false;
        if constexpr (std::is_convertible_v<decltype(k), std::string>) {
          Write(std::string(k));
        } else {
          Write(std::to_string(k));
        }
        buf_ += ':';
        Write(v);
      }
      buf_ += '}';

    } else if constexpr (StructSerializable<T>) {
      buf_ += '{';
      first_field_ = true;
      value.JsonSerialize(*this);
      buf_ += '}';

    } else {
      static_assert(sizeof(T) == 0, "Unsupported type in JsonWriter::Write()");
    }
  }

  //-----------------------------------------------------------------------
  // Object‑field helpers (name + value)
  //-----------------------------------------------------------------------
  template <typename FieldT>
  void Field(std::string_view name, const FieldT& val) {
    if (!first_field_) buf_ += ',';
    first_field_ = false;
    Write(std::string(name));
    buf_ += ':';
    Write(val);
  }

  // Recursive pair‑pack helper – expects even number of args.
  template <typename KeyT, typename ValT, typename... Rest>
  void Fields(const KeyT& key, const ValT& val, const Rest&... rest) {
    Field(key, val);
    if constexpr (sizeof...(rest) > 0) {
      Fields(rest...);
    }
  }

 private:
  std::string& buf_;
  bool first_field_ = true;
};
//==============================================================================
//  Convenience free‑function – mirrors `Serialize()`.
//==============================================================================
template <typename T>
[[nodiscard]] std::string ToJson(const T& obj) {
  std::string out;
  JsonWriter w(out);
  w.Write(obj);
  return out;
}
//==============================================================================
//  Shorthand macros – identical ergonomics to BinaryWriter::WriteFields()
//==============================================================================
#define SETU_JSON_FIELD(writer, field) (writer).Field(#field, field)

#define SETU_JSON_WRITE_FIELDS_1(w, f1) SETU_JSON_FIELD(w, f1)
#define SETU_JSON_WRITE_FIELDS_2(w, f1, f2) \
  SETU_JSON_FIELD(w, f1);                   \
  SETU_JSON_FIELD(w, f2)
#define SETU_JSON_WRITE_FIELDS_3(w, f1, f2, f3) \
  SETU_JSON_FIELD(w, f1);                       \
  SETU_JSON_FIELD(w, f2);                       \
  SETU_JSON_FIELD(w, f3)
#define SETU_JSON_WRITE_FIELDS_4(w, f1, f2, f3, f4) \
  SETU_JSON_FIELD(w, f1);                           \
  SETU_JSON_FIELD(w, f2);                           \
  SETU_JSON_FIELD(w, f3);                           \
  SETU_JSON_FIELD(w, f4)
#define SETU_JSON_WRITE_FIELDS_5(w, f1, f2, f3, f4, f5) \
  SETU_JSON_FIELD(w, f1);                               \
  SETU_JSON_FIELD(w, f2);                               \
  SETU_JSON_FIELD(w, f3);                               \
  SETU_JSON_FIELD(w, f4);                               \
  SETU_JSON_FIELD(w, f5)
#define SETU_JSON_WRITE_FIELDS_6(w, f1, f2, f3, f4, f5, f6) \
  SETU_JSON_FIELD(w, f1);                                   \
  SETU_JSON_FIELD(w, f2);                                   \
  SETU_JSON_FIELD(w, f3);                                   \
  SETU_JSON_FIELD(w, f4);                                   \
  SETU_JSON_FIELD(w, f5);                                   \
  SETU_JSON_FIELD(w, f6)
#define SETU_JSON_WRITE_FIELDS_7(w, f1, f2, f3, f4, f5, f6, f7) \
  SETU_JSON_FIELD(w, f1);                                       \
  SETU_JSON_FIELD(w, f2);                                       \
  SETU_JSON_FIELD(w, f3);                                       \
  SETU_JSON_FIELD(w, f4);                                       \
  SETU_JSON_FIELD(w, f5);                                       \
  SETU_JSON_FIELD(w, f6);                                       \
  SETU_JSON_FIELD(w, f7)
#define SETU_JSON_WRITE_FIELDS_8(w, f1, f2, f3, f4, f5, f6, f7, f8) \
  SETU_JSON_FIELD(w, f1);                                           \
  SETU_JSON_FIELD(w, f2);                                           \
  SETU_JSON_FIELD(w, f3);                                           \
  SETU_JSON_FIELD(w, f4);                                           \
  SETU_JSON_FIELD(w, f5);                                           \
  SETU_JSON_FIELD(w, f6);                                           \
  SETU_JSON_FIELD(w, f7);                                           \
  SETU_JSON_FIELD(w, f8)

#define SETU_JSON_ARG_COUNT_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define SETU_JSON_ARG_COUNT(...) \
  SETU_JSON_ARG_COUNT_IMPL(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1)

//----------- two‑step pasting to ensure expansion -------------
#define SETU_JSON_CAT_(a, b) a##b
#define SETU_JSON_CAT(a, b) SETU_JSON_CAT_(a, b)
#define SETU_JSON_SEL(n) SETU_JSON_CAT(SETU_JSON_WRITE_FIELDS_, n)

#define SETU_JSON_WRITE_FIELDS_DISPATCH(w, n, ...) \
  SETU_JSON_SEL(n)(w, __VA_ARGS__)
#define SETU_JSON_WRITE_FIELDS(w, ...)                                 \
  SETU_JSON_WRITE_FIELDS_DISPATCH(w, SETU_JSON_ARG_COUNT(__VA_ARGS__), \
                                  __VA_ARGS__)
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
