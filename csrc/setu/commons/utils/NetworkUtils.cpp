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
#include "commons/utils/NetworkUtils.h"
//==============================================================================
#include "commons/Logging.h"
#include "commons/StdCommon.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
[[nodiscard]] std::string GetLocalIpAddress() {
  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock == -1) {
    return "127.0.0.1";
  }

  struct sockaddr_in serv;
  std::memset(&serv, 0, sizeof(serv));
  serv.sin_family = AF_INET;
  serv.sin_addr.s_addr = inet_addr("10.254.254.254");
  serv.sin_port = htons(1);

  int err = connect(sock, (const struct sockaddr*)&serv, sizeof(serv));
  std::string ip_address = "127.0.0.1";

  if (err != -1) {
    struct sockaddr_in name;
    socklen_t namelen = sizeof(name);
    err = getsockname(sock, (struct sockaddr*)&name, &namelen);
    if (err != -1) {
      char buffer[INET_ADDRSTRLEN];
      const char* p =
          inet_ntop(AF_INET, &name.sin_addr, buffer, INET_ADDRSTRLEN);
      if (p) {
        ip_address = p;
      }
    }
  }

  close(sock);
  return ip_address;
}
//==============================================================================
[[nodiscard]] std::size_t GetRandomPort() {
  // Get current time for seeding
  auto now = std::chrono::high_resolution_clock::now();
  auto seed = now.time_since_epoch().count();

  // Create random engine with the seed
  std::mt19937 rng(static_cast<unsigned int>(seed));

  // Generate random port in the range 8000-65535
  std::uniform_int_distribution<std::size_t> dist(8000, 65535);

  std::size_t port = 0;
  while (!port || IsPortInUse(port)) {
    port = dist(rng);
  }

  return port;
}
//==============================================================================
[[nodiscard]] std::vector<std::size_t> GetRandomPorts(int n) {
  std::vector<std::size_t> ports;
  for (int i = 0; i < n; ++i) {
    std::size_t port = GetRandomPort();
    while (std::find(ports.begin(), ports.end(), port) != ports.end()) {
      port = GetRandomPort();
    }
    ports.push_back(port);
  }
  return ports;
}
//==============================================================================
[[nodiscard]] bool IsPortInUse(std::size_t port) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock == -1) {
    return true;  // Assume port is in use if we can't create a socket
  }

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(port));

  int result = connect(sock, (struct sockaddr*)&addr, sizeof(addr));
  close(sock);

  return result == 0;  // If connect succeeded, port is in use
}
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
