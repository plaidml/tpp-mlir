#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# Run long benchmarks after building TPP-MLIR.
# shellcheck disable=SC1091

SCRIPT_DIR=$(realpath "$(dirname "$0")/..")
source "${SCRIPT_DIR}/ci/common.sh"

BENCH_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH}/benchmarks
CONFIG_DIR=${BENCH_DIR}/config

LOGFILE=$(mktemp)
trap 'rm ${LOGFILE}' EXIT

# Build
eval "${SCRIPT_DIR}/buildkite/build_tpp.sh"

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
           -n 1000 \
           -c "${CONFIG_DIR}/${JSON}" \
           --build "${BUILD_DIR}-${COMPILER}" \
           | tee -a "${LOGFILE}"
  popd || exit 1
}

# OpenMP Benchmarks
benchmark omp/dnn-fp32.json "OpenMP XSMM-DNN FP32"
benchmark omp/dnn-bf16.json "OpenMP XSMM-DNN BF16"
benchmark omp/mlir-fp32.json "OpenMP TPP-MLIR FP32"
benchmark omp/mlir-bf16.json "OpenMP TPP-MLIR BF16"

# Matmul Benchmarks
benchmark matmul/256x1024x1024.json "Matmul 256x1024x1024"
benchmark matmul/256x1024x4096.json "Matmul 256x1024x4096"
benchmark matmul/256x4096x1024.json "Matmul 256x4096x1024"
benchmark matmul/128x1024x4096.json "Matmul 128x1024x4096"
benchmark matmul/128x4096x1024.json "Matmul 128x4096x1024"
benchmark matmul/128x1024x1024.json "Matmul 128x1024x1024"
benchmark matmul/256x768x768.json "Matmul 256x768x768"
benchmark matmul/128x768x768.json "Matmul 128x768x768"
benchmark matmul/128x3072x768.json "Matmul 128x3072x768"
benchmark matmul/128x768x3072.json "Matmul 128x768x3072"
benchmark matmul/256x3072x768.json "Matmul 256x3072x768"
benchmark matmul/256x768x3072.json "Matmul 256x768x3072"
benchmark matmul/128x768x2304.json "Matmul 128x768x2304"
benchmark matmul/1024x2560x1024.json "Matmul 1024x2560x1024"
benchmark matmul/1024x1024x512.json "Matmul 1024x1024x512"
benchmark matmul/1024x352x512.json "Matmul 1024x352x512"
benchmark matmul/1024x512x256.json "Matmul 1024x512x256"

# FC Benchmarks
benchmark fc/256x1024x1024.json "FC 256x1024x1024"
benchmark fc/256x1024x4096.json "FC 256x1024x4096"
benchmark fc/256x4096x1024.json "FC 256x4096x1024"
benchmark fc/128x1024x4096.json "FC 128x1024x4096"
benchmark fc/128x4096x1024.json "FC 128x4096x1024"
benchmark fc/128x1024x1024.json "FC 128x1024x1024"
benchmark fc/256x768x768.json "FC 256x768x768"
benchmark fc/128x768x768.json "FC 128x768x768"
benchmark fc/128x3072x768.json "FC 128x3072x768"
benchmark fc/128x768x3072.json "FC 128x768x3072"
benchmark fc/256x3072x768.json "FC 256x3072x768"
benchmark fc/256x768x3072.json "FC 256x768x3072"
benchmark fc/128x768x2304.json "FC 128x768x2304"
benchmark fc/1024x2560x1024.json "FC 1024x2560x1024"
benchmark fc/1024x1024x512.json "FC 1024x1024x512"
benchmark fc/1024x352x512.json "FC 1024x352x512"
benchmark fc/1024x512x256.json "FC 1024x512x256"

# Models Benchmarks
benchmark models/models.json "MHA self attention full"

# Summary report for all benchmarks
echo "+++ REPORT"
if [ "main" = "${BUILDKITE_BRANCH}" ]; then
  export LOGRPTBRN=main
fi
eval "${LIBXSMMROOT}/scripts/tool_logrept.sh ${LOGFILE}"
