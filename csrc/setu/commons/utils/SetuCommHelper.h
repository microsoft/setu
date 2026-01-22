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
#include "commons/messages/Messages.h"
#include "commons/utils/Serialization.h"
#include "commons/utils/ZmqHelper.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::BinaryBuffer;
using setu::commons::BinaryRange;
using setu::commons::ClientIdentity;
using setu::commons::NonCopyableNonMovable;
using setu::commons::messages::AnyClientRequest;
using setu::commons::messages::AnyCoordinatorRequest;
//==============================================================================
// SetuCommHelper - Static helper for Setu protocol communication over ZMQ
//
// Wire format uses std::variant serialization - the variant index is written
// automatically, eliminating the need for a separate header frame.
//
// Supported socket patterns:
//   - REQ/REP/DEALER: Use Send(), Recv<T>(), TryRecv<T>()
//   - ROUTER: Use SendToClient(), RecvRequestFromClient(), etc.
//==============================================================================
class SetuCommHelper : public NonCopyableNonMovable {
 public:
  //============================================================================
  // REQ/REP/DEALER pattern: Single frame with serialized message
  //============================================================================

  /// @brief Send a typed message (single frame)
  template <typename T>
  static void Send(ZmqSocketPtr socket, const T& message) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);
    SendFrame(socket, message, zmq::send_flags::none);
  }

  /// @brief Receive a typed message (blocking)
  template <typename T>
  [[nodiscard]] static T Recv(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    zmq::message_t msg;
    auto result = socket->recv(msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive message frame");

    return DeserializeFrame<T>(msg);
  }

  /// @brief Try to receive a typed message (non-blocking)
  template <typename T>
  [[nodiscard]] static std::optional<T> TryRecv(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    zmq::message_t msg;
    auto result = socket->recv(msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) {
      return std::nullopt;
    }

    return DeserializeFrame<T>(msg);
  }

  //============================================================================
  // ROUTER pattern for REQ clients: [Identity][Delimiter][Body]
  // REQ sockets add an empty delimiter frame
  //============================================================================

  /// @brief Send a typed response to a specific client (REQ pattern)
  template <typename T>
  static void SendToClient(ZmqSocketPtr socket, const ClientIdentity& identity,
                           const T& message) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);
    ASSERT_VALID_ARGUMENTS(!identity.empty(),
                           "Client identity cannot be empty");

    // Identity frame
    zmq::message_t identity_msg(identity.data(), identity.size());
    auto result = socket->send(identity_msg, zmq::send_flags::sndmore);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send identity frame");

    // Empty delimiter frame (REQ pattern)
    zmq::message_t delimiter_msg;
    result = socket->send(delimiter_msg, zmq::send_flags::sndmore);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send delimiter frame");

    // Body frame
    SendFrame(socket, message, zmq::send_flags::none);
  }

  /// @brief Receive a request from a REQ client (blocking)
  /// @return Tuple of (client identity, request variant)
  [[nodiscard]] static std::tuple<ClientIdentity, AnyClientRequest>
  RecvRequestFromClient(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Identity frame
    zmq::message_t identity_msg;
    auto result = socket->recv(identity_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive identity frame");
    ASSERT_VALID_RUNTIME(identity_msg.more(),
                         "Expected multipart message, but identity was last");

    ClientIdentity identity(static_cast<const char*>(identity_msg.data()),
                            identity_msg.size());

    // Delimiter frame
    zmq::message_t delimiter_msg;
    result = socket->recv(delimiter_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive delimiter frame");
    ASSERT_VALID_RUNTIME(delimiter_msg.more(),
                         "Expected multipart message, but delimiter was last");

    // Body frame - deserialize variant directly
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    return {std::move(identity), DeserializeFrame<AnyClientRequest>(body_msg)};
  }

  /// @brief Try to receive a request from a REQ client (non-blocking)
  [[nodiscard]] static std::optional<
      std::tuple<ClientIdentity, AnyClientRequest>>
  TryRecvRequestFromClient(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Identity frame (non-blocking)
    zmq::message_t identity_msg;
    auto result = socket->recv(identity_msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) {
      return std::nullopt;
    }

    ASSERT_VALID_RUNTIME(identity_msg.more(),
                         "Expected multipart message, but identity was last");

    ClientIdentity identity(static_cast<const char*>(identity_msg.data()),
                            identity_msg.size());

    // Delimiter frame (blocking - we already started receiving)
    zmq::message_t delimiter_msg;
    result = socket->recv(delimiter_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(),
                         "Failed to receive delimiter frame");
    ASSERT_VALID_RUNTIME(delimiter_msg.more(),
                         "Expected multipart message, but delimiter was last");

    // Body frame
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    return std::make_tuple(std::move(identity),
                           DeserializeFrame<AnyClientRequest>(body_msg));
  }

  //============================================================================
  // DEALER pattern: Single frame (no identity/delimiter needed)
  //============================================================================

  /// @brief Receive a coordinator request from a DEALER socket (blocking)
  [[nodiscard]] static AnyCoordinatorRequest RecvCoordinatorRequest(
      ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    zmq::message_t msg;
    auto result = socket->recv(msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive message frame");

    return DeserializeFrame<AnyCoordinatorRequest>(msg);
  }

  /// @brief Try to receive a coordinator request from a DEALER socket
  /// (non-blocking)
  [[nodiscard]] static std::optional<AnyCoordinatorRequest>
  TryRecvCoordinatorRequest(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    zmq::message_t msg;
    auto result = socket->recv(msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) {
      return std::nullopt;
    }

    return DeserializeFrame<AnyCoordinatorRequest>(msg);
  }

  //============================================================================
  // ROUTER pattern for DEALER clients: [Identity][Body]
  // DEALER sockets do NOT add a delimiter frame
  //============================================================================

  /// @brief Try to receive a client request from a DEALER via ROUTER socket
  /// @note DEALER→ROUTER has no delimiter frame, unlike REQ→ROUTER
  [[nodiscard]] static std::optional<
      std::tuple<ClientIdentity, AnyClientRequest>>
  TryRecvRequestFromNodeAgent(ZmqSocketPtr socket) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);

    // Identity frame (non-blocking)
    zmq::message_t identity_msg;
    auto result = socket->recv(identity_msg, zmq::recv_flags::dontwait);
    if (!result.has_value()) {
      return std::nullopt;
    }

    ASSERT_VALID_RUNTIME(identity_msg.more(),
                         "Expected multipart message, but identity was last");

    ClientIdentity identity(static_cast<const char*>(identity_msg.data()),
                            identity_msg.size());

    // Body frame (no delimiter in DEALER→ROUTER pattern)
    zmq::message_t body_msg;
    result = socket->recv(body_msg, zmq::recv_flags::none);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to receive body frame");

    return std::make_tuple(std::move(identity),
                           DeserializeFrame<AnyClientRequest>(body_msg));
  }

  /// @brief Send a response to a DEALER via ROUTER socket
  /// @note No delimiter frame in DEALER→ROUTER pattern
  template <typename T>
  static void SendToNodeAgent(ZmqSocketPtr socket,
                              const ClientIdentity& identity,
                              const T& message) {
    ASSERT_VALID_POINTER_ARGUMENT(socket);
    ASSERT_VALID_ARGUMENTS(!identity.empty(),
                           "Dealer identity cannot be empty");

    // Identity frame
    zmq::message_t identity_msg(identity.data(), identity.size());
    auto result = socket->send(identity_msg, zmq::send_flags::sndmore);
    ASSERT_VALID_RUNTIME(result.has_value(), "Failed to send identity frame");

    // Body frame (no delimiter in DEALER→ROUTER pattern)
    SendFrame(socket, message, zmq::send_flags::none);
  }

 private:
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
