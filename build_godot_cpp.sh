#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

# build_godot_cpp.sh
# Usage: build_godot_cpp.sh [target] [jobs]
# Example: ./build_godot_cpp.sh template_debug 8

cd "$(dirname "$0")/godot-cpp" || exit 1

TARGET="${1:-template_debug}"
JOBS="${2:-$(nproc)}"

echo "========== Building GDExtension (godot-cpp) (Linux) =========="
echo "Target: ${TARGET}"
echo "Jobs: ${JOBS}"
echo "============================================================"

# On Linux we assume clang/llvm is available in PATH
scons platform=linux use_llvm=yes target=${TARGET} -j${JOBS}

if [ $? -eq 0 ]; then
    echo
    echo "Build Success! Output in bin folder."
    cd ..
    exit 0
else
    echo
    echo "Build Failed! Check SCons output above."
    cd ..
    exit 1
fi
