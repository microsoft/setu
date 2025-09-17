#!/bin/bash
# Linting scripts for Setu project

set -e          # Exit on any error
set -o pipefail # Exit on pipe failures

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_script_dir}/utils.sh"

# Initialize CI environment if needed
init_ci_environment

# Python linting functions
lint_isort() {
    log_lint "Linting (isort Check)..."
    isort --check-only --profile black setu test setup.py
}

lint_black() {
    log_lint "Linting (Black Check)..."
    black --check setu test setup.py
}

lint_autoflake() {
    log_lint "Linting (autoflake Check)..."
    # Run autoflake and capture output
    local output
    output=$(autoflake --recursive --remove-all-unused-imports --check setu test setup.py 2>&1)
    local exit_code=$?

    # Filter out "No issues detected!" messages and only show remaining output
    local filtered_output
    filtered_output=$(echo "$output" | grep -v "No issues detected!" || true)

    if [ -n "$filtered_output" ]; then
        echo "$filtered_output"
    fi

    return $exit_code
}

lint_pyright() {
    log_lint "Linting (Pyright Type Check)..."
    if command_exists pyright; then
        pyright
    else
        log_error "pyright not found. Install with: npm install -g pyright"
        return 1
    fi
}

# C++ linting functions
lint_cpplint() {
    log_lint "Linting (cpplint Check)..."
    # Run cpplint and capture output
    local output
    output=$(cpplint \
        --recursive \
        --exclude=csrc/third_party \
        --exclude=csrc/include/setu/kernels \
        --exclude=csrc/setu/kernels \
        --filter="-build/include_what_you_use,-whitespace/parens,-whitespace/braces,-runtime/references,-readability/namespace,-whitespace/indent" \
        csrc 2>&1) || true
    local exit_code=$?

    # Filter out "Done processing" lines, but only show remaining output if it exists
    local filtered_output
    filtered_output=$(echo "$output" | grep -v "^Done processing" || true)

    if [ -n "$filtered_output" ]; then
        echo "$filtered_output"
    fi

    return $exit_code
}

lint_clang_format() {
    log_lint "Linting (clang-format Check)..."
    local cpp_files
    cpp_files=$(find_cpp_files)
    if [ -n "$cpp_files" ]; then
        local errors=0
        while IFS= read -r file; do
            if ! clang-format --dry-run --Werror "$file" >/dev/null 2>&1; then
                log_error "clang-format check failed for: $file"
                errors=$((errors + 1))
            fi
        done <<<"$cpp_files"

        if [ $errors -gt 0 ]; then
            log_error "clang-format check failed for $errors file(s)"
            return 1
        fi
    else
        log_warning "No C++ files found"
    fi
}

# Other linting functions
lint_codespell() {
    log_lint "Linting (codespell Check)..."
    if command_exists codespell; then
        codespell \
            --skip='./csrc/third_party/**,./csrc/test/testdata/**,./build/**,*.log,./env*/**,./docs/_build/**,./docs/doxygen_output/**,./site/**,./test_reports/**,./logs/**' \
            -L inout,deques
    else
        log_error "codespell not found. Install with: pip install codespell"
        return 1
    fi
}

lint_shellcheck() {
    log_lint "Linting (shellcheck Check)..."
    local shell_scripts
    shell_scripts=$(find_shell_scripts)
    if [ -n "$shell_scripts" ]; then
        # shellcheck disable=SC2086
        shellcheck --severity=warning $shell_scripts
    else
        log_warning "No shell scripts found"
    fi
}

lint_cmake() {
    log_lint "Linting (cmake-lint Check)..."
    local cmake_files
    cmake_files=$(find_cmake_files)
    if [ -n "$cmake_files" ]; then
        if command_exists cmake-lint; then
            # shellcheck disable=SC2086
            # Run cmake-lint and filter out verbose output when there are no issues
            local output
            output=$(cmake-lint $cmake_files 2>&1)
            local exit_code=$?

            # Only show output if there are actual lint issues (found lint: with content after it)
            if echo "$output" | grep -q "found lint:" && ! echo "$output" | grep -q "found lint:$"; then
                echo "$output"
            elif [ $exit_code -ne 0 ]; then
                # Show output if exit code is non-zero even if we didn't detect lint issues
                echo "$output"
            fi

            return $exit_code
        else
            log_error "cmake-lint not found. Install with: pip install cmakelang"
            return 1
        fi
    else
        log_warning "No CMake files found"
    fi
}

# Helper function to run a linter and track failures
run_linter() {
    local linter_func=$1
    local check_name=$2
    local -n failed_array=$3

    if ! "$linter_func"; then
        failed_array+=("$check_name")
    fi
}

# Main linting function
lint_all() {
    log_info "Running all linting checks..."

    local failed_checks=()

    # Python linting
    run_linter lint_isort "isort" failed_checks
    run_linter lint_black "black" failed_checks
    run_linter lint_autoflake "autoflake" failed_checks
    run_linter lint_pyright "pyright" failed_checks

    # C++ linting
    run_linter lint_clang_format "clang-format" failed_checks
    run_linter lint_cpplint "cpplint" failed_checks

    # Other linting
    run_linter lint_codespell "codespell" failed_checks
    run_linter lint_shellcheck "shellcheck" failed_checks
    run_linter lint_cmake "cmake-lint" failed_checks

    if [ ${#failed_checks[@]} -eq 0 ]; then
        log_success "All Linting Checks Passed ${PARTY_ICON}"
        return 0
    else
        log_error "${#failed_checks[@]} linting check(s) failed: ${failed_checks[*]}"
        return 1
    fi
}

# Main function
main() {
    case "${1:-all}" in
        isort)
            lint_isort
            ;;
        black)
            lint_black
            ;;
        autoflake)
            lint_autoflake
            ;;
        pyright)
            lint_pyright
            ;;
        cpplint)
            lint_cpplint
            ;;
        clang-format)
            lint_clang_format
            ;;
        codespell)
            lint_codespell
            ;;
        shellcheck)
            lint_shellcheck
            ;;
        cmake-lint)
            lint_cmake
            ;;
        all)
            lint_all
            ;;
        *)
            echo "Usage: $0 {isort|black|autoflake|pyright|cpplint|clang-format|codespell|shellcheck|cmake-lint|all}"
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
