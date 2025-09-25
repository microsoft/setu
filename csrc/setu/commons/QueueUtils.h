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
#include "commons/BoostCommon.h"
#include "commons/Logging.h"
#include "commons/StdCommon.h"
//==============================================================================
namespace setu::commons {
//==============================================================================
using boost::concurrent::queue_op_status;
using boost::concurrent::sync_queue_is_closed;
//==============================================================================
/**
 * @brief Safely tries to pull an element from a boost concurrent priority queue
 * @tparam QueueType The boost concurrent priority queue type
 * @param queue The queue to pull from
 * @return The pulled element, or default-constructed element if queue is empty
 * or closed
 */
template <typename QueueType>
[[nodiscard]] auto TryPullFromQueue(std::shared_ptr<QueueType> queue) ->
    typename QueueType::value_type {
  using T = typename QueueType::value_type;
  T element = T{};

  try {
    if (!queue->empty()) {
      queue_op_status status = queue->try_pull(element);
      if (status == queue_op_status::success) {
        return element;
      }
    }
  } catch (const sync_queue_is_closed&) {
    // Queue is closed - return default constructed element
  }

  return T{};
}
//==============================================================================
/**
 * @brief Safely pushes an element to a boost concurrent priority queue
 * @tparam QueueType The boost concurrent priority queue type
 * @param queue The queue to push to
 * @param element The element to push
 */
template <typename QueueType>
void SafePushToQueue(std::shared_ptr<QueueType> queue,
                     const typename QueueType::value_type& element) {
  try {
    queue->push(element);
  } catch (const sync_queue_is_closed&) {
    // Queue is closed - silently ignore
  }
}
//==============================================================================
/**
 * @brief Gets the highest priority element from two boost priority queues
 * @tparam QueueType The boost concurrent priority queue type
 * @param queue1 First priority queue
 * @param queue2 Second priority queue
 * @return The highest priority element, or default-constructed element if both
 * queues are empty
 *
 * Note: Lower numerical priority values indicate higher actual priority.
 * The element not selected will be pushed back to its original queue.
 */
template <typename QueueType>
[[nodiscard]] auto GetHighestPriorityFromTwoQueues(
    std::shared_ptr<QueueType> queue1, std::shared_ptr<QueueType> queue2) ->
    typename QueueType::value_type {
  using T = typename QueueType::value_type;

  // Try to get elements from both queues
  T element1 = TryPullFromQueue(queue1);
  T element2 = TryPullFromQueue(queue2);

  // If both queues are empty, return default-constructed element
  if (!element1 && !element2) {
    return T{};
  }

  // If only one queue has an element, return it
  if (!element1) {
    return element2;
  }
  if (!element2) {
    return element1;
  }

  // Both queues have elements - compare priorities
  // Lower numerical priority value means higher actual priority
  if (element1->GetPriority() < element2->GetPriority()) {
    // element1 has higher priority - put element2 back
    SafePushToQueue(queue2, element2);
    return element1;
  } else {
    // element2 has higher or equal priority - put element1 back
    SafePushToQueue(queue1, element1);
    return element2;
  }
}
//==============================================================================
/**
 * @brief Gets the highest priority element from multiple boost priority queues
 * @tparam QueueType The boost concurrent priority queue type
 * @param queues Vector of queues
 * @return The highest priority element, or default-constructed element if all
 * queues are empty
 *
 * Note: Lower numerical priority values indicate higher actual priority.
 * Elements not selected will be pushed back to their original queues.
 */
template <typename QueueType>
[[nodiscard]] auto GetHighestPriorityFromQueues(
    const std::vector<std::shared_ptr<QueueType>>& queues) ->
    typename QueueType::value_type {
  using T = typename QueueType::value_type;

  if (queues.empty()) {
    return T{};
  }

  // Try to pull from all queues
  std::vector<std::pair<T, std::size_t>> pulled_elements;
  pulled_elements.reserve(queues.size());

  for (std::size_t i = 0; i < queues.size(); ++i) {
    T element = TryPullFromQueue(queues[i]);
    if (element) {
      pulled_elements.emplace_back(element, i);
    }
  }

  // If no elements were pulled, return default-constructed element
  if (pulled_elements.empty()) {
    return T{};
  }

  // Find the element with highest priority (lowest numerical value)
  auto highest_priority_it =
      std::min_element(pulled_elements.begin(), pulled_elements.end(),
                       [](const auto& a, const auto& b) {
                         return a.first->GetPriority() < b.first->GetPriority();
                       });

  T highest_priority_element = highest_priority_it->first;
  std::size_t selected_queue_index = highest_priority_it->second;

  // Push back all other elements to their original queues
  for (const auto& [element, queue_index] : pulled_elements) {
    if (queue_index != selected_queue_index) {
      SafePushToQueue(queues[queue_index], element);
    }
  }

  return highest_priority_element;
}
//==============================================================================
/**
 * @brief Gets the lowest priority element from a boost priority queue by
 * draining and filtering
 * @tparam QueueType The boost concurrent priority queue type
 * @param queue The priority queue to search
 * @return The lowest priority element, or default-constructed element if queue
 * is empty
 *
 * Note: Higher numerical priority values indicate lower actual priority.
 * All elements except the selected one will be pushed back to the queue.
 * This is useful for preemption scenarios where you need the least important
 * element.
 */
template <typename QueueType>
[[nodiscard]] auto GetLowestPriorityFromQueue(std::shared_ptr<QueueType> queue)
    -> typename QueueType::value_type {
  using T = typename QueueType::value_type;

  // Collect all elements from the queue
  std::vector<T> all_elements;
  T element;

  while (!queue->empty()) {
    element = TryPullFromQueue(queue);
    if (element) {
      all_elements.push_back(element);
    } else {
      break;
    }
  }

  if (all_elements.empty()) {
    return T{};
  }

  // Find the element with lowest priority (highest numerical value)
  auto lowest_priority_it = std::max_element(
      all_elements.begin(), all_elements.end(), [](const T& a, const T& b) {
        return a->GetPriority() < b->GetPriority();
      });

  T lowest_priority_element = *lowest_priority_it;

  // Push back all other elements to the queue
  for (const T& elem : all_elements) {
    if (elem != lowest_priority_element) {
      SafePushToQueue(queue, elem);
    }
  }

  return lowest_priority_element;
}
//==============================================================================
}  // namespace setu::commons
//==============================================================================
