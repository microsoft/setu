# Define common interface library for all targets
add_library(setu_common INTERFACE)
target_include_directories(
  setu_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/csrc/ ${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu
                        ${cppzmq_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(
  setu_common
  INTERFACE ${TORCH_LIBRARIES}
            Boost::thread
            Boost::uuid
            Boost::heap
            Boost::container_hash
            Boost::stacktrace
            Boost::dynamic_bitset
            libzmq-static
            dl
            backtrace
            setu_python)

# Add NCCL support (required)
if(NOT NCCL_FOUND)
  message(FATAL_ERROR "NCCL is required but was not found. "
                      "Set NCCL_ROOT, CUDA_HOME, or CUDA_PATH environment variable.")
endif()
target_include_directories(setu_common INTERFACE ${NCCL_INCLUDE_DIRS})
target_link_libraries(setu_common INTERFACE ${NCCL_LIBRARIES})
message(STATUS "NCCL support enabled")

# Function to configure common target properties
function(setu_target_config target_name is_module)
  target_link_libraries(${target_name} PRIVATE setu_common)
  if(NOT ${is_module})
    set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
  if(SETU_GPU_ARCHES)
    set_target_properties(${target_name} PROPERTIES ${SETU_GPU_LANG}_ARCHITECTURES
                                                    "${SETU_GPU_ARCHES}")
  endif()
  target_compile_options(
    ${target_name} PRIVATE $<$<COMPILE_LANGUAGE:${SETU_GPU_LANG}>:${SETU_GPU_FLAGS} -fPIC>
                           $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  target_compile_definitions(${target_name} PRIVATE "-DTORCH_EXTENSION_NAME=${target_name}")
endfunction()

# Function to define Python extension modules with object library reuse
function(define_setu_extension name sources object_libs libs)
  python_add_library(${name} MODULE "${sources}" WITH_SOABI)
  if(object_libs)
    foreach(obj_lib ${object_libs})
      target_sources(${name} PRIVATE $<TARGET_OBJECTS:${obj_lib}>)
    endforeach()
  endif()
  target_link_libraries(${name} PRIVATE ${libs})
  setu_target_config(${name} TRUE)
endfunction()

# Function to define static libraries with object library reuse
function(define_setu_static name sources object_libs libs)
  add_library(${name} STATIC ${sources})
  if(object_libs)
    foreach(obj_lib ${object_libs})
      target_sources(${name} PRIVATE $<TARGET_OBJECTS:${obj_lib}>)
    endforeach()
  endif()
  target_link_libraries(${name} PRIVATE ${libs})
  setu_target_config(${name} FALSE)
endfunction()

# OPTIMIZATION: Create OBJECT libraries to compile common sources only once
file(GLOB_RECURSE COMMON_SRC "csrc/setu/commons/*.cpp")

# Create object library for common sources (compiled only once!)
add_library(setu_common_objects OBJECT ${COMMON_SRC})
target_link_libraries(setu_common_objects PRIVATE setu_common)
set_target_properties(setu_common_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(setu_common_objects PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)

# Create object library for CUDA kernels
file(GLOB_RECURSE KERNEL_COMMON_SRC "csrc/setu/kernels/*.cu")
add_library(setu_kernels_cuda_objects OBJECT ${KERNEL_COMMON_SRC})
target_link_libraries(setu_kernels_cuda_objects PRIVATE setu_common)
set_target_properties(setu_kernels_cuda_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(SETU_GPU_ARCHES)
  set_target_properties(setu_kernels_cuda_objects PROPERTIES ${SETU_GPU_LANG}_ARCHITECTURES
                                                             "${SETU_GPU_ARCHES}")
endif()
target_compile_options(setu_kernels_cuda_objects
                       PRIVATE $<$<COMPILE_LANGUAGE:${SETU_GPU_LANG}>:${SETU_GPU_FLAGS} -fPIC>)

# Define kernel common static library using object libraries
add_library(_kernels_common STATIC)
target_sources(_kernels_common PRIVATE $<TARGET_OBJECTS:setu_kernels_cuda_objects>
                                       $<TARGET_OBJECTS:setu_common_objects>)
setu_target_config(_kernels_common FALSE)

# Define specific targets using object libraries
file(GLOB_RECURSE KERNELS_SRC "csrc/setu/kernels/*.cpp")
define_setu_extension(_kernels "${KERNELS_SRC}" "setu_common_objects" "_kernels_common")
define_setu_static(_kernels_static "${KERNELS_SRC}" "setu_common_objects" "_kernels_common")

define_setu_extension(_commons "csrc/setu/commons/Pybind.cpp" "setu_common_objects" "")
define_setu_static(_commons_static "" "setu_common_objects" "")

file(GLOB_RECURSE CLIENT_SRC "csrc/setu/client/*.cpp")
define_setu_extension(_client "${CLIENT_SRC}" "setu_common_objects" "")
define_setu_static(_client_static "${CLIENT_SRC}" "setu_common_objects" "")

# Create object library for IR instruction sources (shared between node_manager, coordinator, and
# _ir)
file(GLOB_RECURSE IR_INSTR_SRC "csrc/setu/ir/instructions/*.cpp")
list(APPEND IR_INSTR_SRC "csrc/setu/ir/Instruction.cpp")
add_library(setu_ir_objects OBJECT ${IR_INSTR_SRC})
target_link_libraries(setu_ir_objects PRIVATE setu_common)
set_target_properties(setu_ir_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(setu_ir_objects PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)

# IR extension module (setu._ir)
define_setu_extension(_ir "csrc/setu/ir/Pybind.cpp" "setu_common_objects;setu_ir_objects" "")

file(GLOB_RECURSE NODE_MANAGER_SRC "csrc/setu/node_manager/*.cpp")
define_setu_extension(_node_manager "${NODE_MANAGER_SRC}" "setu_common_objects;setu_ir_objects"
                      "_kernels_common")
define_setu_static(_node_manager_static "${NODE_MANAGER_SRC}" "setu_common_objects;setu_ir_objects"
                   "_kernels_common")

file(GLOB_RECURSE COORDINATOR_SRC "csrc/setu/coordinator/*.cpp")
list(FILTER COORDINATOR_SRC EXCLUDE REGEX ".*/instructions/.*\\.cpp$")
list(FILTER COORDINATOR_SRC EXCLUDE REGEX ".*/datatypes/Instruction\\.cpp$")
define_setu_extension(_coordinator "${COORDINATOR_SRC}" "setu_common_objects;setu_ir_objects" "")
define_setu_static(_coordinator_static "${COORDINATOR_SRC}" "setu_common_objects;setu_ir_objects"
                   "")

# OPTIMIZATION: Enhanced build graph and parallel compilation
set_target_properties(setu_common_objects PROPERTIES INTERPROCEDURAL_OPTIMIZATION
                                                     $<$<CONFIG:Release>:ON>)

# Link-time optimizations for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set_target_properties(
    _kernels _commons
    PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON LINK_WHAT_YOU_USE ON # Remove unused
               # libraries
  )
endif()

# Parallel compilation hints for ninja - optimized for high-thread systems
include(ProcessorCount)
ProcessorCount(N)
if(N GREATER 16)
  set(CMAKE_JOB_POOLS "compile=${N};link=4") # Use all cores, limit linking
elseif(N GREATER 8)
  math(EXPR COMPILE_JOBS "${N} * 3 / 4")
  set(CMAKE_JOB_POOLS "compile=${COMPILE_JOBS};link=3")
else()
  set(CMAKE_JOB_POOLS "compile=8;link=2") # Default for smaller systems
endif()
message(STATUS "Build parallelism: ${CMAKE_JOB_POOLS}")

# OPTIMIZATION: Apply module-specific precompiled headers after targets are defined
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16)
  # Commons PCH - lightweight, basic headers
  target_precompile_headers(
    setu_common_objects PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/commons/PrecompiledCommonHeaders.h")
  # Kernels PCH - CUDA and kernel-specific headers
  target_precompile_headers(
    setu_kernels_cuda_objects PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/kernels/PrecompiledKernelHeaders.h")

  # Commons PCH - comprehensive headers for commons C++ code
  target_precompile_headers(
    _commons PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/commons/PrecompiledCommonHeaders.h")
  target_precompile_headers(
    _commons_static PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/commons/PrecompiledCommonHeaders.h")

  target_precompile_headers(
    _kernels PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/kernels/PrecompiledKernelHeaders.h")
  target_precompile_headers(
    _kernels_static PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/kernels/PrecompiledKernelHeaders.h")

  target_precompile_headers(
    _client PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/client/PrecompiledCommonHeaders.h")
  target_precompile_headers(
    _client_static PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/client/PrecompiledCommonHeaders.h")

  target_precompile_headers(
    _node_manager PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/node_manager/PrecompiledCommonHeaders.h")
  target_precompile_headers(
    _node_manager_static PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/node_manager/PrecompiledCommonHeaders.h")

  target_precompile_headers(
    _coordinator PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/coordinator/PrecompiledCommonHeaders.h")
  target_precompile_headers(
    _coordinator_static PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu/coordinator/PrecompiledCommonHeaders.h")

  message(STATUS "Module-specific precompiled headers enabled")
endif()
