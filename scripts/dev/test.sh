#!/bin/bash
# Testing scripts for Setu project

set -e          # Exit on any error
set -o pipefail # Exit on pipe failures

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_script_dir}/utils.sh"
source "${_script_dir}/build.sh"

# Initialize CI environment if needed
init_ci_environment

# Python test functions
test_pyunit() {
    log_test "Running Python Unit Tests..."
    setup_dirs

    build_smart

    local timestamp
    timestamp=$(get_timestamp)
    log_info "Test log: logs/pytest_unit_${timestamp}.log"

    pytest -m "unit" \
        --junitxml=test_reports/pytest-unit-results.xml \
        --cov=setu \
        --cov-report=xml:test_reports/python_coverage.xml \
        --cov-report=html:test_reports/python_coverage_html \
        2>&1 | tee "logs/pytest_unit_${timestamp}.log"

    log_success "Python unit tests complete"
}

test_pyintegration() {
    log_test "Running Python Integration Tests..."
    setup_dirs

    build_smart

    local timestamp
    timestamp=$(get_timestamp)
    log_info "Test log: logs/pytest_integration_${timestamp}.log"

    # pytest -m "integration" \
    #     --junitxml=test_reports/pytest-integration-results.xml \
    #     --cov=setu --cov-append \
    #     --cov-report=xml:test_reports/python_coverage.xml \
    #     --cov-report=html:test_reports/python_coverage_html \
    #     2>&1 | tee "logs/pytest_integration_${timestamp}.log"

    log_warning "Python integration tests are not implemented yet"

    log_success "Python integration tests complete"
}

# C++ test functions
test_ctest() {
    log_test "Running C++ tests (ctest) with reports..."
    setup_dirs

    local build_type build_subdir timestamp torch_lib_dir
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"
    timestamp=$(get_timestamp)

    # Ensure tests are built
    log_info "Building native tests first..."
    build_smart_test

    # Setup PyTorch library path
    torch_lib_dir=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
    export LD_LIBRARY_PATH="$torch_lib_dir:${LD_LIBRARY_PATH:-}"

    log_info "Test log: logs/ctest_${timestamp}.log"

    # Get absolute path to test_reports from project root
    local project_root
    project_root="$(cd "$(dirname "$0")/../.." && pwd)"

    # Run ctest with enhanced configuration
    (cd "$build_subdir" \
        && ctest -R "kernel_tests|native_tests" \
            --output-on-failure \
            --timeout 180 \
            --test-dir . \
            --output-junit "$project_root/test_reports/ctest-results.xml") 2>&1 | tee "logs/ctest_${timestamp}.log"

    log_success "C++ tests complete"
}

# Functional test aggregates
test_functional() {
    log_test "Running all functional tests..."

    log_warning "Functional tests are not implemented yet"

    log_success "All functional tests passed"
}

# Unit test aggregates
test_unit() {
    log_test "Running all unit tests (Python + C++)..."

    test_pyunit
    test_ctest

    log_success "All unit tests passed"
}

# Failed-only test functions
test_pyunit_failed_only() {
    log_test "Running only failed Python unit tests from last run..."
    setup_dirs

    build_smart

    local timestamp
    timestamp=$(get_timestamp)
    log_info "Test log: logs/pytest_unit_failed_${timestamp}.log"

    # Check if pytest cache exists
    if [[ ! -d ".pytest_cache" ]]; then
        log_warning "No pytest cache found. Running all Python unit tests instead..."
        test_pyunit
        return
    fi

    # Run only failed tests from last run
    pytest -m "unit" --lf \
        --junitxml=test_reports/pytest-unit-failed-results.xml \
        --cov=setu \
        --cov-report=xml:test_reports/python_coverage_failed.xml \
        --cov-report=html:test_reports/python_coverage_failed_html \
        2>&1 | tee "logs/pytest_unit_failed_${timestamp}.log"

    log_success "Failed Python unit tests rerun complete"
}

test_ctest_failed_only() {
    log_test "Running only failed C++ tests from last ctest run..."
    setup_dirs

    local build_type build_subdir timestamp torch_lib_dir
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"
    timestamp=$(get_timestamp)

    # Ensure tests are built
    log_info "Building native tests first..."
    build_smart_test

    # Setup PyTorch library path
    torch_lib_dir=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
    export LD_LIBRARY_PATH="$torch_lib_dir:${LD_LIBRARY_PATH:-}"

    log_info "Test log: logs/ctest_failed_${timestamp}.log"

    # Get absolute path to test_reports from project root
    local project_root
    project_root="$(cd "$(dirname "$0")/../.." && pwd)"

    # Check if CTest has been run before and has failure information
    if [[ ! -f "$build_subdir/Testing/Temporary/LastTestsFailed.log" ]]; then
        log_warning "No previously failed C++ tests found. Running all C++ tests instead..."
        test_ctest
        return
    fi

    # Run only failed tests from last run
    log_info "Rerunning previously failed C++ tests..."
    (cd "$build_subdir" \
        && ctest --rerun-failed \
            --output-on-failure \
            --timeout 180 \
            --test-dir . \
            --output-junit "$project_root/test_reports/ctest-failed-results.xml") 2>&1 | tee "logs/ctest_failed_${timestamp}.log"

    log_success "Failed C++ tests rerun complete"
}

test_unit_failed_only() {
    log_test "Running only failed unit tests from last run (Python + C++)..."

    test_pyunit_failed_only
    test_ctest_failed_only

    log_success "All failed unit tests rerun complete"
}

# Integration test aggregates
test_integration() {
    log_test "Running all integration tests..."

    test_pyintegration

    log_success "All integration tests passed"
}

# All tests
test_all() {
    log_test "Running all tests..."

    test_unit
    test_integration
    test_functional

    log_success "All tests passed ${PARTY_ICON}"
}

# Main function
main() {
    case "${1:-all}" in
        pyunit)
            test_pyunit
            ;;
        pyintegration)
            test_pyintegration
            ;;
        ctest)
            test_ctest
            ;;
        functional)
            test_functional
            ;;
        unit)
            test_unit
            ;;
        unit-failed-only)
            test_unit_failed_only
            ;;
        pyunit-failed-only)
            test_pyunit_failed_only
            ;;
        ctest-failed-only)
            test_ctest_failed_only
            ;;
        integration)
            test_integration
            ;;
        all)
            test_all
            ;;
        *)
            echo "Usage: $0 {pyunit|pyintegration|ctest|performance|correctness|functional|unit|unit-failed-only|pyunit-failed-only|ctest-failed-only|integration|all}"
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
