#!/usr/bin/env bash
#
# This script is meant to be used in a new machine, to build LLVM, TPP-MLIR
# and run all benchmarks. This should work in the same way as our local tests
# and reproduce our numbers on local/cloud machines.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

# Install packages needed
if [ "$(is_linux_distro Ubuntu)" == "YES" ]; then
  sudo apt update && \
  sudo apt install -y build-essential \
                      cmake clang lld ninja-build \
                      unzip python3-pip libomp-dev git
elif [ "$(is_linux_distro Amazon)" == "YES" ]; then
  sudo dnf install -y cmake clang lld ninja-build \
                      unzip python3-pip libomp-devel git
else
  echo "Not Ubuntu distro, tools may not be available"
fi

# Environment used by the scripts
SOURCE_DIR=$(git_root)
export KIND=Release
export COMPILER=clang
export LINKER=lld

# Build LLVM
export LLVMROOT=${HOME}/installs/llvm
export LLVM_VERSION=$(llvm_version)
export LLVM_INSTALL_DIR=${LLVMROOT}/${LLVM_VERSION}
export LLVM_TAR_DIR=${SOURCE_DIR}/llvm
export LLVM_BUILD_DIR=${SOURCE_DIR}/llvm/build
if [ ! -f "${LLVM_INSTALL_DIR}/bin/mlir-opt" ]; then
  ${SCRIPT_DIR}/github/build_llvm.sh
else
  echo "LLVM already built on ${LLVM_INSTALL_DIR}"
fi

# Build TPP-MLIR
export BUILDKITE_BUILD_CHECKOUT_PATH=${SOURCE_DIR}
export BUILD_DIR=${SOURCE_DIR}/build-${COMPILER}
${SCRIPT_DIR}/github/build_tpp.sh

# Run benchmarks
export BENCH_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH:-.}/benchmarks
export CONFIG_DIR=$(realpath "${BENCH_DIR}/config")
export NUM_ITER=${NUM_ITER:=1000}

pushd ${BENCH_DIR}

echo " ========= Base Benchmarks ==========="
echo_run ./driver.py -vv \
         -n ${NUM_ITER} \
         -c "${CONFIG_DIR}/base/base.json" \
         --build "${BUILD_DIR}"

echo " ========= PyTorch Benchmarks ==========="
echo_run ./driver.py -vv \
         -n ${NUM_ITER} \
         -c "${CONFIG_DIR}/pytorch/torch_dynamo.json" \
         --build "${BUILD_DIR}"

echo " ========= OpenMP Benchmarks ==========="
for cfg in dnn-fp32 dnn-bf16 mlir-fp32 mlir-bf16; do
  echo_run ./driver.py -vv \
           -n ${NUM_ITER} \
           -c "${CONFIG_DIR}/omp/${cfg}.json" \
           --build "${BUILD_DIR}"
done

popd
