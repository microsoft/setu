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
#include "commons/StdCommon.h"
//==============================================================================
/**
 * @namespace setu::commons
 * @brief Common types and type aliases used throughout the Setu system
 *
 * This namespace contains fundamental type definitions used across the entire
 * Setu codebase, including time types, identifier types, tensor-related types,
 * and binary serialization types.
 */
namespace setu::commons {
//==============================================================================
// Time related types
/// @brief Time duration in milliseconds (floating point for precision)
using TimeMS = double;
/// @brief Time duration in seconds (floating point for precision)
using TimeS = double;

// Identifier types for distributed system components
/// @brief Unique identifier for a node in the distributed system
using NodeRank = std::size_t;
/// @brief Unique identifier for a client connection
using ClientRank = std::size_t;
/// @brief Global identifier for a device across all nodes
using DeviceRank = std::size_t;
/// @brief Local identifier for a device within a single node
using LocalDeviceRank = std::size_t;
/// @brief Unique serial number for tracking objects
using SerialNumber = std::uint64_t;
/// @brief Unique identity for client identity (zmq)
using Identity = std::string;

// Tensor related types
/// @brief Name identifier for a tensor
using TensorName = std::string;
/// @brief Name identifier for a tensor dimension
using TensorDimName = std::string;
/// @brief Generic pointer to device memory location
using DevicePtr = void*;
/// @brief Index type for addressing tensor elements (signed for negative
/// indexing)
using TensorIndex = std::int64_t;
/// @brief Set of tensor indices for sparse selections
using TensorIndices = std::set<TensorIndex>;
/// @brief Shared pointer to a set of tensor indices
using TensorIndicesPtr = std::shared_ptr<TensorIndices>;
/// @brief Efficient bitset representation for large index sets
using TensorIndicesBitset = boost::dynamic_bitset<>;
/// @brief Map from dimension names to their corresponding index bitsets
using TensorIndicesMap = std::unordered_map<TensorDimName, TensorIndicesBitset>;
/// @brief Unique identifier for a tensor shard (UUID)
using ShardId = boost::uuids::uuid;
/// @brief Unique identifier for a copy operation (UUID)
using CopyOperationId = boost::uuids::uuid;
/// @brief Unique identifier for a request (UUID)
using RequestId = boost::uuids::uuid;

// Binary serialization related types
/// @brief Buffer for storing binary serialized data
using BinaryBuffer = std::vector<std::uint8_t>;
/// @brief Const iterator for reading binary data
using BinaryIterator = BinaryBuffer::const_iterator;
/// @brief Range specification for binary data segments
using BinaryRange = std::pair<BinaryIterator, BinaryIterator>;
//==============================================================================
}  // namespace setu::commons
//==============================================================================
