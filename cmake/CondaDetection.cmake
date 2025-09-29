# ##################################################################################################
#
# Copyright (C) 2025 by Georgia Institute of Technology
#
# This file is part of the setu project.
#
# ##################################################################################################
# @file                       CondaDetection.cmake @author                     Vajra Team
# <agawalamey12@gmail.com> @date                       25th Jan 2025
# ##################################################################################################

# Robust conda environment detection using SETU_PYTHON_EXECUTABLE Every conda environment has a
# conda-meta directory, so we use that as the key indicator

# Detects conda environment from SETU_PYTHON_EXECUTABLE and sets up include/lib paths. This function
# checks if the Python executable is in a conda environment by looking for the conda-meta directory.
# If found, it sets up the necessary include and library directories for the build system. Sets the
# following parent scope variables: CONDA_PREFIX - Path to the conda environment root
# CONDA_ENVIRONMENT_DETECTED - TRUE if conda environment was found, FALSE otherwise
function(detect_and_setup_conda)
  set(conda_detected FALSE)
  set(conda_prefix_path "")

  # Check if SETU_PYTHON_EXECUTABLE is defined and points to a valid Python executable
  if(DEFINED SETU_PYTHON_EXECUTABLE AND EXISTS "${SETU_PYTHON_EXECUTABLE}")
    # Get the directory containing the Python executable (should be bin/)
    get_filename_component(PYTHON_BIN_DIR "${SETU_PYTHON_EXECUTABLE}" DIRECTORY)

    # Get the parent directory (should be the conda environment root)
    get_filename_component(POTENTIAL_CONDA_PREFIX "${PYTHON_BIN_DIR}" DIRECTORY)

    # Check if conda-meta directory exists (definitive conda environment indicator)
    if(EXISTS "${POTENTIAL_CONDA_PREFIX}/conda-meta" AND IS_DIRECTORY
                                                         "${POTENTIAL_CONDA_PREFIX}/conda-meta")
      set(conda_prefix_path "${POTENTIAL_CONDA_PREFIX}")
      set(conda_detected TRUE)
      message(STATUS "Conda environment detected via SETU_PYTHON_EXECUTABLE: ${conda_prefix_path}")

      # Verify and add include directory
      if(EXISTS "${conda_prefix_path}/include" AND IS_DIRECTORY "${conda_prefix_path}/include")
        include_directories("${conda_prefix_path}/include")
        message(STATUS "Added conda include directory: ${conda_prefix_path}/include")
      else()
        message(WARNING "Conda include directory not found: ${conda_prefix_path}/include")
      endif()

      # Verify and add lib directory
      if(EXISTS "${conda_prefix_path}/lib" AND IS_DIRECTORY "${conda_prefix_path}/lib")
        link_directories("${conda_prefix_path}/lib")
        message(STATUS "Added conda lib directory: ${conda_prefix_path}/lib")
      else()
        message(WARNING "Conda lib directory not found: ${conda_prefix_path}/lib")
      endif()

      # Also check lib64 on systems that use it (common on Linux)
      if(EXISTS "${conda_prefix_path}/lib64" AND IS_DIRECTORY "${conda_prefix_path}/lib64")
        link_directories("${conda_prefix_path}/lib64")
        message(STATUS "Added conda lib64 directory: ${conda_prefix_path}/lib64")
      endif()

      # Set variables for other CMake files to use
      set(CONDA_PREFIX
          "${conda_prefix_path}"
          PARENT_SCOPE)
      set(CONDA_ENVIRONMENT_DETECTED
          TRUE
          PARENT_SCOPE)
    else()
      message(STATUS "SETU_PYTHON_EXECUTABLE does not point to a conda environment: "
                     "${SETU_PYTHON_EXECUTABLE}")
    endif()
  else()
    message(STATUS "SETU_PYTHON_EXECUTABLE not defined or invalid: ${SETU_PYTHON_EXECUTABLE}")
  endif()

  if(NOT conda_detected)
    message(STATUS "No conda environment detected - using system libraries")
    set(CONDA_ENVIRONMENT_DETECTED
        FALSE
        PARENT_SCOPE)
  endif()
endfunction()

detect_and_setup_conda()
