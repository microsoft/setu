if(NOT SETU_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "Please set SETU_PYTHON_EXECUTABLE to the desired Python executable.")
endif()

message(STATUS "SETU_PYTHON_EXECUTABLE: ${SETU_PYTHON_EXECUTABLE}")

find_python_from_executable(${SETU_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
find_package(Python REQUIRED COMPONENTS Development.Module Development.Embed)

message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")

# Python::Module — for extension .so files (no libpython link; symbols come
#   from the host interpreter at load time).
# Python::Python — for standalone executables (test binaries) that embed the
#   interpreter and therefore need libpython.
add_library(setu_python INTERFACE)
target_include_directories(setu_python INTERFACE ${Python_INCLUDE_DIRS})
target_link_libraries(setu_python INTERFACE Python::Module)

add_library(setu_python_embed INTERFACE)
target_include_directories(setu_python_embed INTERFACE ${Python_INCLUDE_DIRS})
target_link_libraries(setu_python_embed INTERFACE Python::Python)

# If using a conda environment, sometimes we need to explicitly add the lib path
if(DEFINED ENV{CONDA_PREFIX})
  message(STATUS "Conda environment detected: $ENV{CONDA_PREFIX}")
  target_include_directories(
    setu_python
    INTERFACE $ENV{CONDA_PREFIX}/include/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})
  target_link_directories(setu_python INTERFACE $ENV{CONDA_PREFIX}/lib)
endif()
