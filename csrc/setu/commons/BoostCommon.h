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
// Boost headers
#include <boost/algorithm/hex.hpp>
#include <boost/container_hash/hash.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/stacktrace.hpp>
#include <boost/thread/concurrent_queues/queue_op_status.hpp>
#include <boost/thread/concurrent_queues/sync_priority_queue.hpp>
#include <boost/thread/concurrent_queues/sync_queue.hpp>
#include <boost/uuid/detail/md5.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
//==============================================================================
namespace setu::commons {
//==============================================================================
template <typename... Args>
using Queue = ::boost::concurrent::sync_queue<Args...>;
template <typename... Args>
using PriorityQueue = ::boost::concurrent::sync_priority_queue<Args...>;
//==============================================================================
/**
 * @brief Generate a random UUID
 *
 * This function provides a layer of indirection over Boost UUID implementation,
 * allowing the main codebase to avoid direct Boost symbol usage.
 *
 * @return A randomly generated UUID
 */
[[nodiscard]] inline boost::uuids::uuid GenerateUUID() {
  static boost::uuids::random_generator generator;
  return generator();
}
//==============================================================================
template <typename T>
struct is_boost_queue : std::false_type {};

template <typename T>
struct is_boost_queue<boost::sync_queue<T>> : std::true_type {};

template <typename T, typename Comp>
struct is_boost_queue<boost::sync_priority_queue<T, Comp>> : std::true_type {};

template <typename T, typename Container, typename Comp>
struct is_boost_queue<boost::sync_priority_queue<T, Container, Comp>>
    : std::true_type {};
//==============================================================================
}  // namespace setu::commons
//==============================================================================
