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
#include "commons/ZmqCommon.h"
//==============================================================================
#include "commons/ClassTraits.h"
#include "commons/Logging.h"
#include "commons/StdCommon.h"
#include "commons/utils/Serialization.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::NonCopyableNonMovable;
//==============================================================================
// Forward declarations and type aliases
using ZmqSocketPtr = std::shared_ptr<zmq::socket_t>;
using ZmqContextPtr = std::shared_ptr<zmq::context_t>;
//==============================================================================
class ZmqHelper : public NonCopyableNonMovable {
 public:
  /**
   * @brief Send a setu object over ZMQ socket using custom serialization
   *
   * @tparam T The setu type to send (must satisfy Serializable concept)
   * @param socket The ZMQ socket to send over
   * @param obj The setu object to send
   */
  template <Serializable T>
  static void Send(ZmqSocketPtr socket, const T& obj) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    const auto serialized_data = Serialize(obj);

    zmq::message_t message(serialized_data.size());
    std::memcpy(message.data(), serialized_data.data(), serialized_data.size());

    const auto result = socket->send(message, zmq::send_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to send serialized message of size {}",
                         serialized_data.size());
  }

  /**
   * @brief Receive a setu object over ZMQ socket using custom serialization
   *
   * @tparam T The setu type to receive (must satisfy Serializable concept)
   * @param socket The ZMQ socket to receive from
   * @return T The received setu object
   */
  template <Serializable T>
  [[nodiscard]] static T Recv(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    zmq::message_t message;
    const auto result = socket->recv(message, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive serialized message");

    const auto* data = static_cast<const std::uint8_t*>(message.data());
    const std::span<const std::uint8_t> message_span(data, message.size());

    return Deserialize<T>(message_span);
  }

  /**
   * @brief Try to send a setu object over ZMQ socket with non-blocking mode
   *
   * @tparam T The setu type to send (must satisfy Serializable concept)
   * @param socket The ZMQ socket to send over
   * @param obj The setu object to send
   * @return true if successful, false otherwise
   */
  template <Serializable T>
  [[nodiscard]] static bool TrySend(ZmqSocketPtr socket, const T& obj) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    try {
      const auto serialized_data = Serialize(obj);

      zmq::message_t message(serialized_data.size());
      std::memcpy(message.data(), serialized_data.data(),
                  serialized_data.size());

      const auto result = socket->send(message, zmq::send_flags::dontwait);
      return result.has_value();
    } catch (const std::exception& e) {
      LOG_WARNING("Failed to send message: {}", e.what());
      return false;
    }
  }

  /**
   * @brief Try to receive a setu object over ZMQ socket with non-blocking mode
   *
   * @tparam T The setu type to receive (must satisfy Serializable concept)
   * @param socket The ZMQ socket to receive from
   * @return Optional containing the received object, or nullopt if no message
   * available
   */
  template <Serializable T>
  [[nodiscard]] static std::optional<T> TryRecv(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    try {
      zmq::message_t message;
      const auto result = socket->recv(message, zmq::recv_flags::dontwait);

      if (!result.has_value()) {
        return std::nullopt;
      }

      const auto* data = static_cast<const std::uint8_t*>(message.data());
      const std::span<const std::uint8_t> message_span(data, message.size());

      return Deserialize<T>(message_span);
    } catch (const std::exception& e) {
      LOG_WARNING("Failed to receive message: {}", e.what());
      return std::nullopt;
    }
  }

  /**
   * @brief Send multiple objects as a multipart message
   *
   * @tparam T The setu type to send (must satisfy Serializable concept)
   * @param socket The ZMQ socket to send over
   * @param objects Vector of objects to send
   */
  template <Serializable T>
  static void SendMultipart(ZmqSocketPtr socket,
                            const std::vector<T>& objects) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);
    ASSERT_VALID_ARGUMENTS(!objects.empty(),
                           "Cannot send empty multipart message");

    for (std::size_t i = 0; i < objects.size(); ++i) {
      const auto serialized_data = Serialize(objects[i]);

      zmq::message_t message(serialized_data.size());
      std::memcpy(message.data(), serialized_data.data(),
                  serialized_data.size());

      const bool is_last = (i == objects.size() - 1);
      const auto flags =
          is_last ? zmq::send_flags::none : zmq::send_flags::sndmore;

      const auto result = socket->send(message, flags);
      ASSERT_VALID_RUNTIME(result.has_value(),
                           "Failed to send multipart message part {}", i);
    }
  }

  /**
   * @brief Receive multiple objects from a multipart message
   *
   * @tparam T The setu type to receive (must satisfy Serializable concept)
   * @param socket The ZMQ socket to receive from
   * @return Vector of received objects
   */
  template <Serializable T>
  [[nodiscard]] static std::vector<T> RecvMultipart(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    std::vector<T> results;

    while (true) {
      zmq::message_t message;
      const auto result = socket->recv(message, zmq::recv_flags::none);
      ASSERT_VALID_RUNTIME(result.has_value(),
                           "Failed to receive multipart message");

      const auto* data = static_cast<const std::uint8_t*>(message.data());
      const std::span<const std::uint8_t> message_span(data, message.size());

      results.push_back(Deserialize<T>(message_span));

      if (!message.more()) {
        break;
      }
    }

    return results;
  }

 private:
  ZmqHelper() = delete;
  ~ZmqHelper() = delete;
};
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
