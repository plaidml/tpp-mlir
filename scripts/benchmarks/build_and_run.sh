#!/usr/bin/env bash
#
# This script is meant to be used in a new machine, to build LLVM, TPP-MLIR
# and run all benchmarks. This should work in the same way as our local tests
# and reproduce our numbers on local/cloud machines.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

# Install packages needed (conda)
CONDA_DIR="$HOME/conda"
CONDA_INIT="${CONDA_DIR}/bin/init"
ARCH_NAME=$(uname -m)
if [ ! -d "${CONDA_DIR}" ]; then
  echo "Installing Conda into ${CONDA_DIR}"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH_NAME}.sh
  bash Miniconda3-latest-Linux-${ARCH_NAME}.sh -b -p "${CONDA_DIR}"
  # Init Conda avoiding polluting .bashrc
  echo "Initializing Conda"
  RC="$HOME/.bashrc"
  mv ${RC} ${RC}.tmp
  touch ${RC}
  conda init
  mv ${RC} ${CONDA_INIT}
  mv ${RC}.tmp ${RC}
fi

echo "Running Conda hook"
eval "$(conda shell.bash hook)"
echo "Running Conda init"
source ${CONDA_INIT}
echo "Running Conda activate"
conda activate
echo "Running Conda install packages"
conda install -y cmake ninja git clang clangxx llvm lld llvm-openmp llvm-tools binutils unzip
if [ "${ARCH_NAME}" == "aarch64" ]; then
   conda install -y gcc_linux-aarch64 gxx_linux-aarch64
elif [ "${ARCH_NAME}" == "x86_64" ]; then
   conda install -y gcc_linux-64 gxx_linux-64
fi
python -m pip install coloredlogs

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
  ${SCRIPT_DIR}/buildkite/build_llvm.sh
else
  echo "LLVM already built on ${LLVM_INSTALL_DIR}"
fi

# Build TPP-MLIR
export BUILDKITE_BUILD_CHECKOUT_PATH=${SOURCE_DIR}
export BUILD_DIR=${SOURCE_DIR}/build-${COMPILER}
${SCRIPT_DIR}/buildkite/build_tpp.sh

# Run benchmarks
export BUILDKITE_BENCHMARK_NUM_ITER=1000
export BENCH_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH:-.}/benchmarks
export CONFIG_DIR=$(realpath "${BENCH_DIR}/config")
export NUM_ITER=1000

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
conda deactivate
