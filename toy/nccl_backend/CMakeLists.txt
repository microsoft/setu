add_executable(toy_nccl_backend main.cpp node_agent.cpp planner.cpp worker.cpp)

find_package(spdlog REQUIRED)

find_path(NCCL_INCLUDE_DIR NAMES "nccl.h")
find_library(NCCL_LIBRARY NAMES "nccl")

target_link_libraries(toy_nccl_backend PRIVATE
    CUDA::cudart
    ${NCCL_LIBRARY}
    spdlog::spdlog
)

target_include_directories(toy_nccl_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NCCL_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)