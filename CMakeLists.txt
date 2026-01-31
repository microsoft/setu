# ##################################################################################################
# Copyright (C) 2025 by Vajra Team, Georgia Institute of Technology; Microsoft Corporation.
# This file is part of the setu project.
# ##################################################################################################
cmake_minimum_required(VERSION 4.0)
set(CMAKE_POLICY_VERSION_MINIMUM 4.0)
project(setu_extensions LANGUAGES CXX CUDA)
cmake_policy(SET CMP0077 NEW)
# ##################################################################################################
# Set C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS
    "-g -Wall -Wextra -fvisibility=hidden -pipe -fdiagnostics-color=always -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1 -DBOOST_STACKTRACE_USE_BACKTRACE)

# Build optimization options
option(SETU_ENABLE_CCACHE "Enable ccache for faster compilation" ON)
# Use mold linker for faster linking
set(CMAKE_LINKER_TYPE MOLD)
# ##################################################################################################
# Define supported versions
set(PYTHON_SUPPORTED_VERSIONS "3.12")
set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")
set(TORCH_SUPPORTED_VERSION_CUDA "2.8")
# ##################################################################################################
# Include utility functions and setup modules
include(cmake/Utils.cmake)
include(cmake/CondaDetection.cmake)
include(cmake/CcacheSetup.cmake)
include(cmake/Dependencies.cmake)
include(cmake/PythonSetup.cmake)
include(cmake/TorchSetup.cmake)
include(cmake/ParallelismSetup.cmake)
include(cmake/Extensions.cmake)
include(cmake/Testing.cmake)
# ##################################################################################################
# Define default target
add_custom_target(
  default
  DEPENDS _kernels
          _commons
          _kernels_static
          _commons_static
          _client
          _client_static
          _ir
          _node_manager
          _node_manager_static
          _coordinator
          _coordinator_static
  COMMENT "Build all default targets")
# ##################################################################################################
# Enable testing
enable_testing()
# ##################################################################################################
