#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

# 用法：./build_all.sh [target] [jobs]
# 示例：./build_all.sh template_debug 8

cd "$(dirname "$0")"

TARGET="${1:-template_debug}"
JOBS="${2:-$(nproc)}"
BUILD_OPTIONS=()
if [[ "${TARGET}" == "template_debug" ]]; then
    BUILD_OPTIONS+=(dev_build=yes optimize=debug debug_symbols=yes)
fi

echo "========== Building MiniBrainGd (Linux) =========="
echo "Target: ${TARGET}"
echo "Jobs: ${JOBS}"
echo "Options: ${BUILD_OPTIONS[*]:-(release defaults)}"
echo "=================================================="

# 主 SConstruct 会先增量构建 MiniMind 和 MNN，再链接 GDExtension。
scons platform=linux use_llvm=yes target="${TARGET}" "${BUILD_OPTIONS[@]}" -j"${JOBS}"

echo
echo "Build Success! Output in bin folder."
