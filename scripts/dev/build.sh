#!/bin/bash
# Build scripts for Setu project

set -e          # Exit on any error
set -o pipefail # Exit on pipe failures

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_script_dir}/utils.sh"
source "${_script_dir}/build_env.sh"
source "${_script_dir}/build_detection.sh"

# Initialize CI environment if needed
init_ci_environment

# Get number of parallel build jobs
get_parallel_jobs() {
    # Respect CMAKE_BUILD_PARALLEL_LEVEL or MAX_JOBS if set
    if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL}" ]; then
        echo "${CMAKE_BUILD_PARALLEL_LEVEL}"
    elif [ -n "${MAX_JOBS}" ]; then
        echo "${MAX_JOBS}"
    else
        # Default to number of CPU cores
        nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1
    fi
}

# CMake configuration
cmake_configure() {
    log_build "Configuring CMake..."
    create_build_dirs

    local build_type timestamp
    build_type=$(get_build_type)
    timestamp=$(get_timestamp)

    local build_subdir
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"
    local cmake_args=()

    log_info "Build Type=${YELLOW}${build_type}${RESET}"
    log_info "Build Directory=${YELLOW}${build_subdir}${RESET}"
    log_info "Full log: logs/cmake_configure_${timestamp}.log"

    # Build configuration
    cmake_args+=(
        "-G" "Ninja"
        "-DCMAKE_BUILD_TYPE=${build_type}"
        "-DSETU_PYTHON_EXECUTABLE=$(which python)"
        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(pwd)/setu"
    )

    # Compiler launcher configuration
    if command_exists sccache; then
        cmake_args+=(
            "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache"
        )
    elif command_exists ccache; then
        cmake_args+=(
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
        )
    fi

    # Verbose output if requested
    if [[ "${VERBOSE:-0}" == "1" ]]; then
        cmake_args+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
    fi

    # Configure CMake in subdirectory
    mkdir -p "$build_subdir"
    local source_dir
    source_dir="$(realpath "$(pwd)")"
    (cd "$build_subdir" && cmake "${cmake_args[@]}" "$source_dir") 2>&1 | tee "logs/cmake_configure_${timestamp}.log"

    # Update file list cache after successful configure
    update_file_list_cache "$build_subdir/build.ninja"

    log_success "CMake configuration complete"
}

# Build functions
build_native() {
    log_build "Building native extension (full build)..."
    log_info "CI context: ${SETU_IS_CI_CONTEXT:-0}"
    log_info "Python location: $(which python)"

    cmake_configure

    local build_type build_subdir timestamp
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"
    timestamp=$(get_timestamp)

    log_info "Build log: logs/cmake_build_native_${timestamp}.log"
    (cd "$build_subdir" && cmake --build . --target default --parallel "$(get_parallel_jobs)") 2>&1 | tee "logs/cmake_build_native_${timestamp}.log"
    log_success "Native extension build complete"
}

build_native_incremental() {
    log_build "Building native extension (incremental build)..."
    create_build_dirs

    local build_type build_subdir timestamp
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"
    timestamp=$(get_timestamp)

    if [ ! -f "$build_subdir/build.ninja" ]; then
        log_warning "No existing build found, running full build..."
        build_native
        return
    fi

    log_info "Build log: logs/cmake_build_native_incr_${timestamp}.log"
    (cd "$build_subdir" && cmake --build . --target default --parallel "$(get_parallel_jobs)") 2>&1 | tee "logs/cmake_build_native_incr_${timestamp}.log"
    log_success "Native extension incremental build complete"
}

build_native_test_full() {
    log_build "Building native tests (full build)..."

    cmake_configure

    local build_type build_subdir timestamp
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"
    timestamp=$(get_timestamp)

    log_info "Build log: logs/cmake_build_tests_${timestamp}.log"
    (cd "$build_subdir" && cmake --build . --target all_tests --parallel "$(get_parallel_jobs)") 2>&1 | tee "logs/cmake_build_tests_${timestamp}.log"

    log_success "Native tests build complete"
}

build_native_test_incremental() {
    log_build "Building native tests (incremental build)..."
    create_build_dirs

    local build_type build_subdir timestamp
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"
    timestamp=$(get_timestamp)

    if [ ! -f "$build_subdir/build.ninja" ]; then
        log_warning "No existing build found, running full build..."
        build_smart_test
        return
    fi

    log_info "Build log: logs/cmake_build_tests_incr_${timestamp}.log"
    (cd "$build_subdir" && cmake --build . --target all_tests --parallel "$(get_parallel_jobs)") 2>&1 | tee "logs/cmake_build_tests_incr_${timestamp}.log"
    log_success "Native tests incremental build complete"
}

build_wheel() {
    log_build "Building Python wheel and sdist..."
    create_build_dirs

    # Ensure dependencies are installed first
    "$(dirname "$0")/setup.sh" dependencies

    local cuda_version torch_version timestamp
    cuda_version=$(get_cuda_version)
    torch_version=$(get_torch_version)
    timestamp=$(get_timestamp)

    log_info "Using CUDA=${YELLOW}${cuda_version}${BLUE}, Torch=${YELLOW}${torch_version}${RESET}"

    # Clean previous builds
    rm -rf dist/ build/ *.egg-info/

    # Export environment variables for setup.py
    export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(get_parallel_jobs)}"
    export VERBOSE="${VERBOSE:-0}"

    # Handle build isolation (wheels typically need isolation)
    local build_args=("--wheel" "--sdist")
    if [[ "${USE_BUILD_ISOLATION:-1}" == "0" ]]; then
        log_info "Build isolation disabled"
        build_args+=("--no-isolation")
    fi

    # Build wheel and sdist with logging
    python -m build "${build_args[@]}" 2>&1 | tee "logs/wheel_build_${timestamp}.log"

    # Validate that no unexpected dynamic dependencies leaked in
    source "$(dirname "$0")/validate_build.sh"
    validate_so_deps setu wheel

    log_success "Wheel and sdist built successfully"
    ls -la dist/
}

build_editable() {
    log_build "Installing editable package..."

    # Ensure dependencies are installed first
    "$(dirname "$0")/setup.sh" dependencies

    local cuda_version torch_version timestamp build_type
    cuda_version=$(get_cuda_version)
    torch_version=$(get_torch_version)
    timestamp=$(get_timestamp)
    build_type=$(get_build_type)

    log_info "Using CUDA=${YELLOW}${cuda_version}${BLUE}, Torch=${YELLOW}${torch_version}${RESET}"
    log_info "Build Type=${YELLOW}${build_type}${RESET}"

    # First, build the native extension using our build system
    log_info "Building native extension first..."
    build_native

    # Export environment variables for setup.py
    export SETU_SKIP_CPP_BUILD="1" # Skip C++ build in setup.py since we already built it

    # Handle build isolation
    local pip_args=()
    if [[ "${USE_BUILD_ISOLATION:-0}" == "1" ]]; then
        log_info "Build isolation enabled for CI/production"
        pip_args+=("-e" ".")
    else
        log_info "Build isolation disabled"
        pip_args+=("-e" "." "--no-build-isolation")
    fi

    # Install with logging (skipping C++ build since we already did it)
    log_info "Installing Python package (skipping C++ build)..."
    pip install "${pip_args[@]}" 2>&1 | tee "logs/pip_install_editable_${timestamp}.log"

    # Create pip install marker after successful installation
    create_pip_install_marker

    log_success "Editable package installed successfully"
}

# Intelligent build - automatically determines and executes the optimal build strategy
build_smart() {
    log_build "Intelligent build - analyzing requirements..."

    # log CI context
    log_info "CI context: ${SETU_IS_CI_CONTEXT:-0}"

    local strategy
    strategy=$(determine_build_strategy)

    log_info "${YELLOW}Strategy:${RESET} $strategy"

    # Extract strategy type from the message
    local strategy_type
    if [[ "$strategy" == *"full_install"* ]]; then
        strategy_type="full_install"
    elif [[ "$strategy" == *"native_build"* ]]; then
        strategy_type="native_build"
    elif [[ "$strategy" == *"incremental_build"* ]]; then
        strategy_type="incremental_build"
    else
        log_error "Unknown build strategy: $strategy"
        return 1
    fi

    case "$strategy_type" in
        full_install)
            log_info "Running: ${CYAN}make build${RESET} (editable install)"
            build_editable
            ;;
        native_build)
            log_info "Running: ${CYAN}make build/native${RESET} (native build)"
            build_native
            ;;
        incremental_build)
            log_info "Running: ${CYAN}make build/native_incremental${RESET} (incremental)"
            build_native_incremental
            ;;
    esac

    log_success "Intelligent build completed successfully"
}

# Intelligent test build - automatically determines and executes the optimal test build strategy
build_smart_test() {
    log_build "Intelligent test build - analyzing requirements..."

    local strategy
    strategy=$(determine_test_build_strategy)

    log_info "${YELLOW}Test Strategy:${RESET} $strategy"

    # Extract strategy type from the message
    local strategy_type
    if [[ "$strategy" == *"full_test_build"* ]]; then
        strategy_type="full_test_build"
    elif [[ "$strategy" == *"incremental_test_build"* ]]; then
        strategy_type="incremental_test_build"
    else
        log_error "Unknown test build strategy: $strategy"
        return 1
    fi

    case "$strategy_type" in
        full_test_build)
            log_info "Running: ${CYAN}make build/native_test_full${RESET} (full test build)"
            build_native_test_full
            ;;
        incremental_test_build)
            log_info "Running: ${CYAN}make build/native_test_incremental${RESET} (incremental test build)"
            build_native_test_incremental
            ;;
    esac

    log_success "Intelligent test build completed successfully"
}

# Clean functions
clean_build() {
    local force=false
    if [[ "$1" == "--force" ]]; then
        force=true
    fi

    log_clean "Cleaning Project Artifacts..."

    # Get directories to clean
    local ramdisk_dir
    ramdisk_dir=$(get_ramdisk_config)

    if use_ramdisk; then
        log_warning "This will remove: build/ ($ramdisk_dir) dist/ test_reports/ logs/ *.so *.egg-info coverage files python cache..."
    else
        log_warning "This will remove: build/ dist/ test_reports/ logs/ *.so *.egg-info coverage files python cache..."
    fi

    if ! $force; then
        read -p "Are you sure? [y/N] " -r confirm
        if [[ ! $confirm =~ ^[Yy]([Ee][Ss])?$ ]]; then
            log_info "Aborted."
            return 1
        fi
    fi

    # Clean RAM disk (build directory)
    if use_ramdisk; then
        if [ -d "$ramdisk_dir" ]; then
            log_info "Cleaning build RAM disk: $ramdisk_dir"
            rm -rf "${ramdisk_dir:?}/"* 2>/dev/null || true
        fi
    fi

    # Clean build artifacts
    rm -rf build/ \
        dist/ \
        setu/*.so \
        *.egg-info \
        test_reports/ \
        logs/ \
        .coverage* \
        coverage.xml \
        python_coverage.xml \
        python_coverage_html \
        .pytest_cache \
        .pyright_cache

    log_success "Cleanup completed"
}

# ccache functions
ccache_stats() {
    log_info "Showing ccache statistics..."
    if command_exists ccache; then
        ccache -s
    else
        log_error "ccache not found"
        return 1
    fi
}

ccache_clear() {
    log_info "Clearing ccache..."
    if command_exists ccache; then
        ccache -C
        log_success "ccache cleared"
    else
        log_error "ccache not found"
        return 1
    fi
}

# Main function
main() {
    case "${1:-}" in
        smart)
            build_smart
            ;;
        smart-test)
            build_smart_test
            ;;
        native)
            build_native
            ;;
        native-incremental)
            build_native_incremental
            ;;
        native-test)
            build_native_test_full
            ;;
        native-test-incremental)
            build_native_test_incremental
            ;;
        wheel)
            build_wheel
            ;;
        editable)
            build_editable
            ;;
        clean)
            clean_build
            ;;
        clean-force)
            clean_build --force
            ;;
        ccache-stats)
            ccache_stats
            ;;
        ccache-clear)
            ccache_clear
            ;;
        *)
            echo "Usage: $0 {smart|smart-test|native|native-incremental|native-test|native-test-incremental|wheel|editable|clean|ccache-stats|ccache-clear}"
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
