#!/bin/bash
# Validate that built .so files only depend on expected dynamic libraries.
# Usage: validate_build.sh <so_dir> [wheel|dev]
#   wheel (default): strict — no libpython (extensions loaded by interpreter)
#   dev:             allows libpython (test binaries embed the interpreter)

set -e
set -o pipefail

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_script_dir}/logging.sh"

# Dynamic libraries that are acceptable at runtime. Anything not matching
# these patterns is a build error — it must be statically linked or added here.
ALLOWED_PATTERNS=(
    # System / glibc
    'libc\.so'
    'libm\.so'
    'libpthread\.so'
    'librt\.so'
    'libdl\.so'
    'ld-linux-x86-64\.so'
    'linux-vdso\.so'

    # C++ runtime
    'libstdc\+\+\.so'
    'libgcc_s\.so'

    # OpenMP
    'libgomp\.so'

    # glibc utility
    'libutil\.so'

    # CUDA
    'libcuda'
    'libnv'
    'libcufft'
    'libcublas'
    'libcusparse'
    'libcudnn'
    'libcupti'
    'libcufile'
    'libcurand'

    # PyTorch
    'libtorch'
    'libc10'
    'libshm\.so'

    # NCCL
    'libnccl\.so'

    # FFI
    'libffi\.so'
)

# Additional patterns allowed only in dev/editable builds (standalone test
# binaries embed the interpreter and therefore link libpython).
DEV_EXTRA_PATTERNS=(
    'libpython'
)

validate_so_deps() {
    local so_dir="${1:-.}"
    local mode="${2:-wheel}"  # "wheel" (default) or "dev"
    local so_files=()
    local violations=()

    while IFS= read -r -d '' f; do
        so_files+=("$f")
    done < <(find "$so_dir" -maxdepth 1 -name '*.so' -print0 2>/dev/null)

    if [[ ${#so_files[@]} -eq 0 ]]; then
        log_error "No .so files found in $so_dir"
        return 1
    fi

    log_info "Validating ${#so_files[@]} .so files in $so_dir (mode=$mode)"

    local all_patterns=("${ALLOWED_PATTERNS[@]}")
    if [[ "$mode" == "dev" ]]; then
        all_patterns+=("${DEV_EXTRA_PATTERNS[@]}")
    fi

    local allowed_regex
    allowed_regex=$(printf "|%s" "${all_patterns[@]}")
    allowed_regex="${allowed_regex:1}"

    for so in "${so_files[@]}"; do
        local basename
        basename=$(basename "$so")

        local bad_deps
        bad_deps=$(ldd "$so" 2>/dev/null \
            | grep '=>' \
            | grep -v 'not found' \
            | awk '{print $1}' \
            | grep -Ev "$allowed_regex" \
            || true)

        if [[ -n "$bad_deps" ]]; then
            while IFS= read -r dep; do
                violations+=("$basename -> $dep")
                log_error "  $basename: unexpected dependency $dep"
            done <<< "$bad_deps"
        fi
    done

    if [[ ${#violations[@]} -gt 0 ]]; then
        echo ""
        log_error "Found ${#violations[@]} unexpected dynamic dependencies."
        log_error "These must be statically linked or added to the allowlist in $0"
        return 1
    fi

    log_success "All dynamic dependencies are in the allowlist"
    return 0
}

main() {
    local target_dir="${1:-setu}"
    local mode="${2:-wheel}"  # "wheel" or "dev"
    validate_so_deps "$target_dir" "$mode"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
