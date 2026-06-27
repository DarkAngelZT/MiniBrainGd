#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

# build_all.sh
# Usage: build_all.sh [target] [jobs]
# Example: ./build_all.sh template_debug 8

cd "$(dirname "$0")"

TARGET="${1:-template_debug}"
JOBS="${2:-$(nproc)}"

echo "========== Building MiniBrainGd (Linux) =========="
echo "Target: ${TARGET}"
echo "Jobs: ${JOBS}"
echo "==============================================="

# Invoke SCons to build for Linux. On Linux we assume clang/llvm is
# available in PATH, so we don't require a manual LLVM path input.
scons platform=linux use_llvm=yes target=${TARGET} -j${JOBS}

if [ $? -eq 0 ]; then
    echo
    echo "Build Success! Output in bin folder."
    exit 0
else
    echo
    echo "Build Failed! Check SCons output above."
    exit 1
fi
