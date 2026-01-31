include(FetchContent)

FetchContent_Declare(googletest
                     URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz)

set(BOOST_ENABLE_CMAKE ON)
set(BOOST_INCLUDE_LIBRARIES
    thread
    uuid
    heap
    container_hash
    stacktrace
    dynamic_bitset
    unordered)

FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-cmake.tar.gz)

FetchContent_Declare(
  zmq URL https://github.com/zeromq/libzmq/releases/download/v4.3.4/zeromq-4.3.4.tar.gz)

FetchContent_Declare(cppzmq URL https://github.com/zeromq/cppzmq/archive/refs/tags/v4.10.0.tar.gz)

set(ZMQ_BUILD_TESTS
    OFF
    CACHE BOOL "Build ZeroMQ tests" FORCE)
set(ZMQ_BUILD_DRAFT_API
    OFF
    CACHE BOOL "Build ZeroMQ draft API" FORCE)

FetchContent_MakeAvailable(googletest Boost zmq cppzmq)

# NCCL
if(DEFINED SETU_PYTHON_EXECUTABLE)
  execute_process(
    COMMAND ${SETU_PYTHON_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

set(NCCL_SEARCH_PATHS ${PYTHON_SITE_PACKAGES}/nvidia/nccl /usr/local/cuda)

find_path(
  NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS $ENV{NCCL_ROOT} $ENV{CUDA_HOME} $ENV{CUDA_PATH}
  PATHS ${NCCL_SEARCH_PATHS}
  PATH_SUFFIXES include)

find_library(
  NCCL_LIBRARY
  NAMES nccl
  HINTS $ENV{NCCL_ROOT} $ENV{CUDA_HOME} $ENV{CUDA_PATH}
  PATHS ${NCCL_SEARCH_PATHS}
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR)

if(NCCL_FOUND AND NOT TARGET NCCL::NCCL)
  add_library(NCCL::NCCL UNKNOWN IMPORTED)
  set_target_properties(NCCL::NCCL PROPERTIES IMPORTED_LOCATION "${NCCL_LIBRARY}"
                                              INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
endif()
