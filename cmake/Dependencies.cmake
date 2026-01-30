include(FetchContent)

FetchContent_Declare(googletest
                     URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz)

set(BOOST_ENABLE_CMAKE ON)
set(BOOST_INCLUDE_LIBRARIES thread uuid heap container_hash stacktrace dynamic_bitset)

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
find_path(NCCL_INCLUDE_DIR NAMES "nccl.h")
find_library(NCCL_LIBRARY NAMES "nccl")
