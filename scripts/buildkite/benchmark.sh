#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# Run long benchmarks after building TPP-MLIR.

SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

BENCH_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH}/benchmarks
CONFIG_DIR=${BENCH_DIR}/config

# Build
${SCRIPT_DIR}/buildkite/build_tpp.sh

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
  export LOGFILE=benchmark-output.txt
  pushd ${BENCH_DIR}
  echo_run ./driver.py -v \
           -n 1000 \
           -c ${CONFIG_DIR}/${JSON} \
           --build ${BUILD_DIR}-${COMPILER} \
           | tee ${LOGFILE}
  if [ "main" = "${BUILDKITE_BRANCH}" ]; then
    export LOGRPTBRN=main
  fi
  echo "Benchmark Report"
  cat ${LOGFILE} | ${LIBXSMMROOT}/scripts/tool_logrept.sh
  popd
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
