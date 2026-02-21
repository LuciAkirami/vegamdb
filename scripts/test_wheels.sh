#!/bin/bash
# Test cibuildwheel locally (requires Docker)
# Usage: ./scripts/test_wheels.sh

set -e

echo "=== Testing cibuildwheel locally ==="
echo "Building wheel for CPython 3.10 (manylinux x86_64)..."

CIBW_BUILD="cp310-manylinux_x86_64" \
CIBW_SKIP="*-win32 *-manylinux_i686 *-musllinux*" \
CIBW_BEFORE_BUILD="pip install scikit-build-core pybind11 numpy" \
CIBW_TEST_REQUIRES="pytest numpy" \
CIBW_TEST_COMMAND="python {project}/scripts/run_tests.py {project}" \
cibuildwheel --platform linux --output-dir wheelhouse

echo ""
echo "=== Done! Wheels are in ./wheelhouse/ ==="
ls -lh wheelhouse/*.whl
