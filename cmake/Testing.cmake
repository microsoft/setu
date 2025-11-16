# Define common interface library for tests
add_library(setu_test_common INTERFACE)
target_include_directories(setu_test_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/csrc/setu)
target_link_libraries(setu_test_common INTERFACE setu_common gtest gtest_main)

# Function to configure common test target properties
function(setu_test_config target_name)
  target_link_libraries(${target_name} PRIVATE setu_test_common)
  target_compile_options(${target_name}
                         PRIVATE $<$<COMPILE_LANGUAGE:${SETU_GPU_LANG}>:${SETU_GPU_FLAGS} -fPIC>)
  if(SETU_GPU_ARCHES)
    set_target_properties(${target_name} PROPERTIES ${SETU_GPU_LANG}_ARCHITECTURES
                                                    "${SETU_GPU_ARCHES}")
  endif()
endfunction()

# Define test data copying target
set(TESTDATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc/test/testdata)
add_custom_target(
  copy_testdata
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TESTDATA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/testdata
  COMMENT "Copy test data files")

# Function to add test suites
function(add_setu_test_suite name source_dir libs)
  file(GLOB_RECURSE CPP_TEST_SRC "${source_dir}/*.cpp")
  file(GLOB_RECURSE CUDA_TEST_SRC "${source_dir}/*.cu")
  set(all_test_src ${CPP_TEST_SRC} ${CUDA_TEST_SRC})
  if(all_test_src)
    add_executable(${name}_tests ${all_test_src})
    setu_test_config(${name}_tests)
    target_link_libraries(${name}_tests PRIVATE ${libs})

    # Make test data available before running tests
    add_dependencies(${name}_tests copy_testdata)

    if(CUDA_TEST_SRC)
      set_source_files_properties(${CUDA_TEST_SRC} PROPERTIES LANGUAGE CUDA)
      target_compile_options(
        ${name}_tests PRIVATE $<$<COMPILE_LANGUAGE:${SETU_GPU_LANG}>:${SETU_GPU_FLAGS} -fPIC>
                              $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
      if(SETU_GPU_ARCHES)
        set_target_properties(${name}_tests PROPERTIES ${SETU_GPU_LANG}_ARCHITECTURES
                                                       "${SETU_GPU_ARCHES}")
      endif()
    endif()
    add_test(NAME ${name}_tests COMMAND ${name}_tests
                                        --gtest_output=xml:test_reports/${name}_tests_results.xml)
  endif()
endfunction()

# Add test suites
add_setu_test_suite(kernel "csrc/test/kernels" "_kernels_static")
add_setu_test_suite(native "csrc/test/native" "_commons_static")
# Define all_tests target

add_custom_target(
  all_tests
  DEPENDS default kernel_tests native_tests
  COMMENT "Build all tests")
