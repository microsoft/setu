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
#include "commons/StdCommon.h"
//==============================================================================
namespace setu::commons::utils {
//==============================================================================
using setu::commons::NonCopyableNonMovable;
//==============================================================================
/**
 * A generic registry base class template. Subclasses will have their own
 * registry map. Each subclass can store implementations keyed by an enum type.
 *
 * This is a C++ translation of the Python BaseRegistry class.
 *
 * @tparam EnumType The enum type used as keys
 * @tparam BaseType The base class type of registered implementations
 * @tparam Args Optional parameter types for parameterized factories
 */
template <typename EnumType, typename BaseType, typename... Args>
class BaseRegistry : public NonCopyableNonMovable {
 protected:
  // Factory function type - supports both parameterless and parameterized
  using FactoryFunction = std::function<std::shared_ptr<BaseType>(Args...)>;
  using RegistryMap = std::unordered_map<EnumType, FactoryFunction>;
  static RegistryMap _registry;

 public:
  /**
   * Register an implementation class with the registry.
   *
   * @param key The enum key to register the implementation under
   * @tparam ImplType The implementation class type
   */
  template <typename ImplType>
  static void Register(EnumType key) {
    if (_registry.find(key) != _registry.end()) {
      return;  // Already registered
    }

    _registry[key] = [](Args... args) -> std::shared_ptr<BaseType> {
      return std::make_shared<ImplType>(std::forward<Args>(args)...);
    };
  }

  /**
   * Register an implementation class with the registry with custom factory
   * function.
   *
   * @param key The enum key to register the implementation under
   * @param factory A factory function that creates instances of the
   * implementation
   */
  static void RegisterWithFactory(EnumType key, FactoryFunction factory) {
    if (_registry.find(key) != _registry.end()) {
      return;  // Already registered
    }

    _registry[key] = factory;
  }

  /**
   * Unregister an implementation from the registry.
   *
   * @param key The enum key to unregister
   * @throws std::invalid_argument if the key is not registered
   */
  static void Unregister(EnumType key) {
    if (_registry.find(key) == _registry.end()) {
      throw std::invalid_argument("Key is not registered");
    }

    _registry.erase(key);
  }

  /**
   * Get an instance of the implementation registered under the given key.
   * For parameterless factories (backward compatibility).
   *
   * @param key The enum key to get the implementation for
   * @return A shared pointer to the implementation instance
   * @throws std::invalid_argument if the key is not registered
   */
  [[nodiscard]] static std::shared_ptr<BaseType> Get(EnumType key) {
    static_assert(sizeof...(Args) == 0,
                  "Get() can only be used with parameterless registries. Use "
                  "Create() for parameterized factories.");
    if (_registry.find(key) == _registry.end()) {
      throw std::invalid_argument("Key is not registered");
    }

    return _registry[key]();
  }

  /**
   * Create an instance with the given parameters.
   *
   * @param key The enum key to get the implementation for
   * @param args The arguments to pass to the factory
   * @return A shared pointer to the created instance
   * @throws std::invalid_argument if the key is not registered
   */
  [[nodiscard]] static std::shared_ptr<BaseType> Create(EnumType key,
                                                        Args... args) {
    if (_registry.find(key) == _registry.end()) {
      throw std::invalid_argument("Key is not registered");
    }

    return _registry[key](std::forward<Args>(args)...);
  }

  /**
   * Check if a key is registered.
   *
   * @param key The key to check
   * @return true if registered, false otherwise
   */
  [[nodiscard]] static bool IsRegistered(EnumType key) {
    return _registry.find(key) != _registry.end();
  }

  /**
   * Get all registered keys.
   *
   * @return A vector of all registered keys
   */
  [[nodiscard]] static std::vector<EnumType> GetRegisteredKeys() {
    std::vector<EnumType> keys;
    keys.reserve(_registry.size());
    for (const auto& [key, _] : _registry) {
      keys.push_back(key);
    }
    return keys;
  }
};

// Initialize the static registry map for each template instantiation
template <typename EnumType, typename BaseType, typename... Args>
typename BaseRegistry<EnumType, BaseType, Args...>::RegistryMap
    BaseRegistry<EnumType, BaseType, Args...>::_registry = {};

// Convenience alias for parameterless registries (backward compatibility)
template <typename EnumType, typename BaseType>
using SimpleRegistry = BaseRegistry<EnumType, BaseType>;
//==============================================================================
}  // namespace setu::commons::utils
//==============================================================================
