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
namespace setu::commons {
//==============================================================================
/**
 * @brief Makes a class non-copyable but potentially movable.
 *
 * Protected constructor and destructor to allow derivation
 * and prevent direct instantiation or polymorphic deletion via NonCopyable*.
 */
class NonCopyable {
 protected:
  NonCopyable() = default;
  ~NonCopyable() = default;

 public:
  // Delete copy constructor and copy assignment operator
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;

  // Default move constructor and move assignment operator
  // A class that is NonCopyable can still be movable by default.
  // If you want to prevent this, the derived class must delete them,
  // or you can use NonCopyableNonMovable.
  NonCopyable(NonCopyable&&) = default;
  NonCopyable& operator=(NonCopyable&&) = default;
};
//==============================================================================
/**
 * @brief Makes a class non-movable but potentially copyable.
 *
 * Protected constructor and destructor to allow derivation
 * and prevent direct instantiation or polymorphic deletion via NonMovable*.
 */
class NonMovable {
 protected:
  NonMovable() = default;
  ~NonMovable() = default;

 public:
  // Delete move constructor and move assignment operator
  NonMovable(NonMovable&&) = delete;
  NonMovable& operator=(NonMovable&&) = delete;

  // Default copy constructor and copy assignment operator
  // A class that is NonMovable can still be copyable by default.
  NonMovable(const NonMovable&) = default;
  NonMovable& operator=(const NonMovable&) = default;
};
//==============================================================================
/**
 * @brief Makes a class neither copyable nor movable.
 *
 * This is often desired for classes that manage unique resources
 * like file handles, mutexes, or network sockets directly.
 */
class NonCopyableNonMovable {
 protected:
  NonCopyableNonMovable() = default;
  ~NonCopyableNonMovable() = default;

 public:
  // Delete all copy and move operations
  NonCopyableNonMovable(const NonCopyableNonMovable&) = delete;
  NonCopyableNonMovable& operator=(const NonCopyableNonMovable&) = delete;
  NonCopyableNonMovable(NonCopyableNonMovable&&) = delete;
  NonCopyableNonMovable& operator=(NonCopyableNonMovable&&) = delete;
};
//==============================================================================
/**
 * @brief Trait to make a derived class behave like a "static class"
 * (i.e., uninstantiable).
 */
class StaticClass {
 public:
  // Delete all constructors
  StaticClass() = delete;
  StaticClass(const StaticClass&) = delete;
  StaticClass(StaticClass&&) = delete;

  // Delete assignment operators
  StaticClass& operator=(const StaticClass&) = delete;
  StaticClass& operator=(StaticClass&&) = delete;

  // Delete destructor
  // This is important as it prevents stack allocation if other measures fail
  // and also prevents polymorphic deletion (though not relevant if
  // uninstantiable).
  ~StaticClass() = delete;
};
//==============================================================================
}  // namespace setu::commons
//==============================================================================
