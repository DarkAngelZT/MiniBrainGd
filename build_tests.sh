#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

cd "$(dirname "$0")"

JOBS="${1:-$(nproc)}"

echo "========== Building MiniBrainGd C++ tests =========="
scons -f SConstruct.tests -j"${JOBS}"

echo "========== Running MiniBrainGd C++ tests =========="
./tests/bin/minibrain_gd_tests
