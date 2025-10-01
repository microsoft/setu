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
#include "commons/Types.h"
#include "commons/datatypes/TensorSelection.h"
//==============================================================================
namespace setu::commons::datatypes {
//==============================================================================
/**
 * @brief Specification for copying tensor data between source and destination
 *
 * CopySpec defines a copy operation between two tensors, specifying the source
 * and destination tensor names along with their corresponding selections. The
 * selections must be compatible (same dimensions and sizes) for the copy to be
 * valid.
 */
struct CopySpec {
  /**
   * @brief Constructs a copy specification with source and destination details
   *
   * @param src_name_param Name of the source tensor
   * @param dst_name_param Name of the destination tensor
   * @param src_selection_param Selection of the source tensor to copy from
   * @param dst_selection_param Selection of the destination tensor to copy to
   *
   * @throws std::invalid_argument if either selection is null or if selections
   * are incompatible
   */
  CopySpec(TensorName src_name_param, TensorName dst_name_param,
           TensorSelectionPtr src_selection_param,
           TensorSelectionPtr dst_selection_param)
      : src_name(src_name_param),
        dst_name(dst_name_param),
        src_selection(src_selection_param),
        dst_selection(dst_selection_param) {
    ASSERT_VALID_POINTER_ARGUMENT(src_selection_param);
    ASSERT_VALID_POINTER_ARGUMENT(dst_selection_param);
    ASSERT_VALID_ARGUMENTS(
        src_selection_param->IsCompatible(dst_selection_param),
        "Source and destination selections are not compatible");
  }

  /**
   * @brief Returns a string representation of the copy specification
   *
   * @return String containing source name, destination name, and both
   * selections
   */
  [[nodiscard]] std::string ToString() const {
    return std::format(
        "CopySpec(src_name={}, dst_name={}, src_selection={}, "
        "dst_selection={})",
        src_name, dst_name, src_selection, dst_selection);
  }

  const TensorName src_name;               ///< Name of the source tensor
  const TensorName dst_name;               ///< Name of the destination tensor
  const TensorSelectionPtr src_selection;  ///< Selection from the source tensor
  const TensorSelectionPtr
      dst_selection;  ///< Selection for the destination tensor
};
//==============================================================================
/// @brief Shared pointer to a CopySpec object
using CopySpecPtr = std::shared_ptr<CopySpec>;
//==============================================================================
}  // namespace setu::commons::datatypes
//==============================================================================
