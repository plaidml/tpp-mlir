#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# CMake and build TPP-MLIR.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

LLVMROOT=$(realpath ${LLVMROOT})
if [ ! -d ${LLVMROOT} ]; then
  echo "'${OPTARG}' not a LLVMROOT directory"
  exit 1
fi

# Find LLVM_VERSION
echo "--- LLVM"
LLVM_VERSION=$(llvm_version)
if [ ! -d "${LLVMROOT}/${LLVM_VERSION}" ]; then
  echo "LLVM ${LLVM_VERSION} not found"
  exit 1
else
  echo "Found LLVM ${LLVM_VERSION}"
fi

# CMake
echo "--- CONFIGURE"
if [ ! "${COMPILER}" ]; then
  COMPILER=clang
fi
if [ "${COMPILER}" == "clang" ]; then
  GCC_COMPAT_OPTION="-g ${GCC_TOOLCHAIN}"
fi
if [ "${SANITIZERS}" ]; then
  SANITIZERS="-S"
fi
if [ "${INSTALL}" ]; then
  if [ -d "${INSTALL_PREFIX}" ]; then
    INSTALL_OPTION="-i ${INSTALL_PREFIX}"
  else
    echo "Installation requires INSTALL_PREFIX (=${INSTALL_PREFIX}) to be a valid directory"
    exit 1
  fi
fi
if [ ! "${LINKER}" ]; then
  LINKER=lld
fi
if [ "${GPU}" ]; then
  GPU_OPTION="-G"
fi

BLD_DIR="${BUILD_DIR}-${COMPILER}"
if ! ${SCRIPT_DIR}/ci/cmake.sh \
  -s ${BUILDKITE_BUILD_CHECKOUT_PATH} \
  -b ${BLD_DIR} -R \
  -m ${LLVMROOT}/${LLVM_VERSION}/lib/cmake/mlir \
  ${INSTALL_OPTION} \
  -t ${KIND} \
  ${SANITIZERS} \
  ${GPU_OPTION} \
  -c ${COMPILER} \
  ${GCC_COMPAT_OPTION} \
  -l ${LINKER} \
  -n ${NPROCS_LIMIT_LINK}
then
  exit 1
fi

echo "Configuring support for GPUs"
CUDA_DRIVER=/usr/lib64/libcuda.so
if [ ! -f "${CUDA_DRIVER}" ]; then
  # GPU setting originally had boolean semantic ("cuda" if true)
  if [[ (${GPU} =~ ^[+-]?[0-9]+([.][0-9]+)?$) && ("0" != "${GPU}") ]] || [ "cuda" = "${GPU}" ]; then
    echo "GPU support requires full CUDA driver to be present"
    exit 1
  else
    # create link to CUDA stubs (CUDA incorporated by default)
    echo "Creating links to CUDA stubs"
    ln -s ${CUDATOOLKIT_HOME}/lib64/stubs/libcuda.so ${BLD_DIR}/lib/libcuda.so.1
    ln -s ${BLD_DIR}/lib/libcuda.so.1 ${BLD_DIR}/lib/libcuda.so
    export LD_LIBRARY_PATH=${BLD_DIR}/lib:${LD_LIBRARY_PATH}
    # more detailed support for other GPU runtimes, e.g., "vulkan"
    if [ "${GPU}" ]; then
      echo "Enable general support for GPUs"
    fi
  fi
fi

# Build
echo "--- BUILD"
if ! ${SCRIPT_DIR}/ci/build.sh \
  -b ${BLD_DIR}
then
  exit 1
fi

# Check
if [ "${CHECK}" ]; then
  echo "--- CHECK"
  if ! ${SCRIPT_DIR}/ci/build.sh \
    -b ${BLD_DIR} \
    -c
  then
    exit 1
  fi
fi

# Install
if [ "${INSTALL}" ]; then
  echo "--- INSTALL"
  if ! ${SCRIPT_DIR}/ci/build.sh \
    -b ${BLD_DIR} \
    -i
  then
    exit 1
  fi
fi

# Benchmark
if [ "${BENCH}" ]; then
  echo "--- BENCHMARK"
  export LOGFILE=benchmark-output.txt
  ${SCRIPT_DIR}/ci/build.sh \
    -b ${BLD_DIR} \
    -B | tee ${LOGFILE}
  echo "--- RESULTS"
  if [ "main" = "${BUILDKITE_BRANCH}" ]; then
    export LOGRPTBRN=main
  fi
  cat ${LOGFILE} | ${LIBXSMMROOT}/scripts/tool_logrept.sh
fi
