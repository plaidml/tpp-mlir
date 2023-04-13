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
    echo "Installation requites INSTALL_PREFIX (=${INSTALL_PREFIX}) to be a valid directory"
    exit 1
  fi
fi
if [ ! "${LINKER}" ]; then
  LINKER=lld
fi
if not ${SCRIPT_DIR}/ci/cmake.sh \
  -s ${BUILDKITE_BUILD_CHECKOUT_PATH} \
  -b ${BUILD_DIR}-${COMPILER} \
  -m ${LLVMROOT}/${LLVM_VERSION}/lib/cmake/mlir \
  ${INSTALL_OPTION} \
  -t ${KIND} \
  ${SANITIZERS} \
  -c ${COMPILER} \
  ${GCC_COMPAT_OPTION} \
  -l ${LINKER} \
  -n ${NPROCS_LIMIT_LINK}
then
  exit 1
fi

# Build
echo "--- BUILD"
if not ${SCRIPT_DIR}/ci/build.sh \
  -b ${BUILD_DIR}-${COMPILER}
then
  exit 1
fi

# Check
if [ "${CHECK}" ]; then
  echo "--- CHECK"
  if not ${SCRIPT_DIR}/ci/build.sh \
    -b ${BUILD_DIR}-${COMPILER} \
    -c
  then
    exit 1
  fi
fi

# Install
if [ "${INSTALL}" ]; then
  echo "--- INSTALL"
  if not ${SCRIPT_DIR}/ci/build.sh \
    -b ${BUILD_DIR}-${COMPILER} \
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
    -b ${BUILD_DIR}-${COMPILER} \
    -B | tee ${LOGFILE}
  echo "--- RESULTS"
  if [ "main" = "${BUILDKITE_BRANCH}" ]; then
    export LOGRPTBRN=main
  fi
  cat ${LOGFILE} | ${LIBXSMMROOT}/scripts/tool_logrept.sh
fi
