#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# CMake and build TPP-MLIR.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

LLVMROOT=${HOME}/installs/llvm
if [ ! -d ${LLVMROOT} ]; then
  mkdir -p ${LLVMROOT}
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
if [ "1" == "${SANITIZERS}" ]; then
  SANITIZERS="-S"
fi
if [ "1" == "${INSTALL}" ]; then
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

if [[ (${GPU} =~ ^[+-]?[0-9]+([.][0-9]+)?$) ]]; then
  if [ "0" != "${GPU}" ]; then GPU_OPTION="-G"; fi
elif [ "${GPU}" ]; then
  GPU_OPTION="-G ${GPU}"
fi

if [ "1" == "${CLEAN}" ]; then
  BLD_DIR_RM=-R
fi

# Defaults when lacking CI environment
PROJECT_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH:-.}
BUILD_DIR=${BUILD_DIR:-build-${COMPILER}}

BLD_DIR=$(realpath ${BUILD_DIR})
if ! ${SCRIPT_DIR}/ci/cmake.sh \
  -s ${PROJECT_DIR} \
  -b ${BLD_DIR} ${BLD_DIR_RM} \
  -m ${LLVMROOT}/${LLVM_VERSION}/lib/cmake/mlir \
  ${INSTALL_OPTION} \
  -t ${KIND} \
  ${SANITIZERS} \
  ${GPU_OPTION} \
  -c ${COMPILER} \
  ${GCC_COMPAT_OPTION} \
  -l ${LINKER} \
  -n ${NPROCS_LIMIT_LINK:-1}
then
  exit 1
fi

CUDA_LIBRARY=/usr/lib64/libcuda.so
# CUDA: original GPU setting had boolean semantic (true/non-zero implies "cuda")
if [[ (${GPU} =~ ^[+-]?[0-9]+([.][0-9]+)?$) && ("0" != "${GPU}") ]] || [ "cuda" == "${GPU}" ]; then
  if [ "${CUDA_LIBRARY}" ]; then
    echo "Enabled GPU backend (cuda)"
  else
    echo "CUDA support requires GPU driver to be present"
    exit 1
  fi
else
  # create link to CUDA stubs (CUDA incorporated by default)
  if [ ! -f "${CUDA_LIBRARY}" ]; then
    if [ "${CUDATOOLKIT_HOME}" ]; then
      echo "Creating links to CUDA stubs"
      export LD_LIBRARY_PATH=${BLD_DIR}/lib:${LD_LIBRARY_PATH}
      if [ -d "${CUDATOOLKIT_HOME}/lib64" ]; then
        CUDA_LIBRARY=${CUDATOOLKIT_HOME}/lib64/stubs/libcuda.so
      else
        CUDA_LIBRARY=${CUDATOOLKIT_HOME}/lib/stubs/libcuda.so
      fi
      ln -fs ${CUDA_LIBRARY} ${BLD_DIR}/lib/libcuda.so.1
      ln -fs ${CUDA_LIBRARY} ${BLD_DIR}/lib/libcuda.so
    else
      echo "CUDA stub libraries are needed but CUDATOOLKIT_HOME is not set"
      exit 1
    fi
  fi
  if [ "${GPU}" ]; then
    echo "Enabled GPU backend (${GPU})"
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
if [ "1" == "${CHECK}" ]; then
  echo "--- CHECK"
  if ! ${SCRIPT_DIR}/ci/build.sh \
    -b ${BLD_DIR} \
    -c
  then
    exit 1
  fi
fi

# Install
if [ "1" == "${INSTALL}" ]; then
  echo "--- INSTALL"
  if ! ${SCRIPT_DIR}/ci/build.sh \
    -b ${BLD_DIR} \
    -i
  then
    exit 1
  fi
fi

# Benchmark
if [ "1" == "${BENCH}" ]; then
  echo "--- BENCHMARK"
  export LOGFILE=benchmark-output.txt
  ${SCRIPT_DIR}/ci/build.sh \
    -b ${BLD_DIR} \
    -B | tee ${LOGFILE}
  echo "--- RESULTS"
  if [ "main" == "${BUILDKITE_BRANCH}" ]; then
    export LOGRPTBRN=main
  fi
  cat ${LOGFILE} | ${LIBXSMMROOT}/scripts/tool_logrept.sh
fi
