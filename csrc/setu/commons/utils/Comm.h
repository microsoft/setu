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
#include "commons/ClassTraits.h"
#include "commons/Logging.h"
#include "commons/Types.h"
#include "commons/ZmqCommon.h"
//==============================================================================
#include "commons/utils/Serialization.h"
#include "commons/utils/ZmqHelper.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::BinaryBuffer;
using setu::commons::BinaryRange;
using setu::commons::Identity;
using setu::commons::NonCopyableNonMovable;

class Comm : public NonCopyableNonMovable {
 public:
  /**
   * @brief Send a typed message as a single frame (blocking).
   *
   * Socket pairs: REQ -> ROUTER, REQ -> REP, DEALER -> ROUTER
   *
   * @tparam T The message type to send
   * @param socket [in] The ZMQ socket to send on
   * @param message [in] The message to send
   */
  template <typename T>
  static void Send(ZmqSocketPtr socket, const T& message) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);
    SendFrame(socket, message, zmq::send_flags::none);
  }

  /**
   * @brief Receive a typed message as a single frame (blocking).
   *
   * Socket pairs: REP <- REQ, DEALER <- ROUTER
   *
   * @tparam T The message type to receive
   * @param socket [in] The ZMQ socket to receive from
   * @return The deserialized message
   */
  template <typename T>
  [[nodiscard]] static T Recv(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    zmq::message_t msg;
    auto result = socket->recv(msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive message frame");

    return DeserializeFrame<T>(msg);
  }

  /**
   * @brief Send a typed message with identity routing via ROUTER socket.
   *
   * Socket pairs:
   *   - ROUTER -> REQ (AddDelimiter=true): Frame layout
   * [Identity][Delimiter][Body]
   *   - ROUTER -> DEALER (AddDelimiter=false): Frame layout [Identity][Body]
   *
   * @tparam T The message type to send
   * @tparam AddDelimiter Whether to add empty delimiter frame (true for REQ)
   * @param socket [in] The ROUTER socket to send on
   * @param identity [in] The client identity to route to
   * @param message [in] The message to send
   */
  template <typename T, bool AddDelimiter = true>
  static void SendWithIdentity(ZmqSocketPtr socket, const Identity& identity,
                               const T& message) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);
    ASSERT_VALID_ARGUMENTS(!identity.empty(),
                           "Client identity cannot be empty");

    zmq::message_t identity_msg(identity.data(), identity.size());
    auto result = socket->send(identity_msg, zmq::send_flags::sndmore);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send identity frame");

    if constexpr (AddDelimiter) {
      zmq::message_t delimiter_msg;
      result = socket->send(delimiter_msg, zmq::send_flags::sndmore);
      ASSERT_VALID_RUNTIME(result.has_value(),
                           "Failed to send delimiter frame");
    }

    SendFrame(socket, message, zmq::send_flags::none);
  }

  /**
   * @brief Receive a typed message with identity from a ROUTER socket
   * (blocking).
   *
   * Socket pairs:
   *   - ROUTER <- REQ (ExpectDelimiter=true): Frame layout
   * [Identity][Delimiter][Body]
   *   - ROUTER <- DEALER (ExpectDelimiter=false): Frame layout [Identity][Body]
   *
   * @tparam T The message type to receive
   * @tparam ExpectDelimiter Whether to expect empty delimiter frame (true for
   * REQ)
   * @param socket [in] The ROUTER socket to receive from
   * @return Tuple of (identity, message)
   */
  template <typename T, bool ExpectDelimiter = true>
  [[nodiscard]] static std::tuple<Identity, T> RecvWithIdentity(
      ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    zmq::message_t identity_msg;
    auto result = socket->recv(identity_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive identity frame");
    ASSERT_VALID_RUNTIME(identity_msg.more(),
                         "Expected multipart message, but identity was last");

    Identity identity(static_cast<const char*>(identity_msg.data()),
                      identity_msg.size());

    if constexpr (ExpectDelimiter) {
      zmq::message_t delimiter_msg;
      result = socket->recv(delimiter_msg, zmq::recv_flags::none);
      ASSERT_VALID_RUNTIME(result.has_value(),
                           "Failed to receive delimiter frame");
      ASSERT_VALID_RUNTIME(
          delimiter_msg.more(),
          "Expected multipart message, but delimiter was last");
    }

    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    return {std::move(identity), DeserializeFrame<T>(body_msg)};
  }

  /**
   * @brief Poll multiple sockets for readability.
   *
   * @param sockets [in] Vector of sockets to poll
   * @param timeout_ms [in] Timeout in milliseconds (-1 for infinite)
   * @return Vector of sockets that are ready to read
   */
  [[nodiscard]] static std::vector<ZmqSocketPtr> PollForRead(
      const std::vector<ZmqSocketPtr>& sockets /*[in]*/,
      std::int32_t timeout_ms /*[in]*/) {
    ASSERT_VALID_ARGUMENTS(!sockets.empty(), "Cannot poll empty socket list");

    std::vector<zmq::pollitem_t> poll_items;
    poll_items.reserve(sockets.size());

    for (const auto& socket : sockets) {
      ASSERT_VALID_POINTER_ARGUMENT(socket);
      poll_items.push_back({socket->handle(), 0, ZMQ_POLLIN, 0});
    }

    zmq::poll(poll_items, std::chrono::milliseconds(timeout_ms));

    std::vector<ZmqSocketPtr> ready;
    for (std::size_t i = 0; i < poll_items.size(); ++i) {
      if (poll_items[i].revents & ZMQ_POLLIN) {
        ready.push_back(sockets[i]);
      }
    }

    return ready;
  }

 private:
  /* Serialize and send an object as a single ZMQ frame. */
  template <typename T>
  static void SendFrame(ZmqSocketPtr socket, const T& obj,
                        zmq::send_flags flags) {
    BinaryBuffer buf;
    BinaryWriter writer(buf);
    writer.Write(obj);

    zmq::message_t message(buf.size());
    std::memcpy(message.data(), buf.data(), buf.size());

    const auto result = socket->send(message, flags);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send frame of size {}",
                         buf.size());
  }

  /* Deserialize a ZMQ message frame into a typed object. */
  template <typename T>
  [[nodiscard]] static T DeserializeFrame(const zmq::message_t& msg) {
    const auto* data = static_cast<const std::uint8_t*>(msg.data());
    BinaryBuffer buf(data, data + msg.size());
    BinaryReader reader(BinaryRange{buf.begin(), buf.end()});
    return reader.Read<T>();
  }
};
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
