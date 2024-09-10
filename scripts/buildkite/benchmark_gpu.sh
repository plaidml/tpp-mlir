#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# Run GPU benchmarks after building TPP-MLIR.
# shellcheck disable=SC1091

SCRIPT_DIR=$(realpath "$(dirname "$0")/..")
source "${SCRIPT_DIR}/ci/common.sh"

BENCH_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH:-.}/benchmarks
BUILD_DIR=$(realpath "${BUILD_DIR:-build-${COMPILER}}")
CONFIG_DIR=$(realpath "${BENCH_DIR}/config")

if [ ! -z "${GPU}" ]; then
  GPU_OPTION="${GPU}"
else
  echo "Benchmark GPU target not specified"
  exit 1
fi

# Build
eval "GPU=${GPU_OPTION} ${SCRIPT_DIR}/buildkite/build_tpp.sh"
if [ $? != 0 ]; then
  exit 1
fi

# Benchmark
benchmark () {
  JSON=$1
  if [ ! -f "${CONFIG_DIR}/${JSON}" ]; then
    echo "Cannot find benchmark configuration '${JSON}'"
    exit 1
  fi
  NAME=$2
  if [ ! "${NAME}" ]; then
    echo "Invalid benchmark name '${NAME}'"
    exit 1
  fi

  echo "--- BENCHMARK '${NAME}'"
  pushd "${BENCH_DIR}" || exit 1
  echo_run ./driver.py -v \
           -n 100 \
           -c "${CONFIG_DIR}/${JSON}" \
           --build "${BUILD_DIR}"
  popd || exit 1
}

# CUDA Benchmarks
if [ "${GPU_OPTION}" == "cuda" ]; then
  source /swtools/cuda/latest/cuda_vars.sh
  benchmark GPU/cuda.json "CUDA kernels"
  if [ $? != 0 ]; then
    exit 1
  fi
fi
