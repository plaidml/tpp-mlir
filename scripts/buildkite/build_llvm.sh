#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# CMake and build LLVM.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

LLVMROOT=${HOME}/installs/llvm
if [ ! -d ${LLVMROOT} ]; then
  mkdir -p ${LLVMROOT}
fi

# LLVM setup
echo "--- LLVM"
LLVM_VERSION=${BUILDKITE_COMMIT}
if [ ! "${LLVM_VERSION}" ]; then
  echo "Unknown LLVM_VERSION version"
  exit 1
fi
echo "LLVM version: ${LLVM_VERSION}"

LLVM_INSTALL_DIR=${LLVMROOT}/${LLVM_VERSION}
if [ ! -d ${LLVM_INSTALL_DIR} ]; then
  mkdir -p ${LLVM_INSTALL_DIR}
fi

LLVM_PROJECTS="mlir"
LLVM_TARGETS="host"

if [ ! "${KIND}" ]; then
  KIND=RelWithDebInfo
fi

# Environment setup
echo "--- ENVIRONMENT"
if [ ! "${COMPILER}" ]; then
  COMPILER=clang
fi
if [ "${COMPILER}" == "clang" ]; then
  check_program clang
  check_program clang++
  CC=clang
  CXX=clang++
elif [ "${COMPILER}" == "gcc" ]; then
  check_program gcc
  check_program g++
  CC=gcc
  CXX=g++
else
  echo "Compiler "${COMPILER}" not recognized"
  exit 1
fi

if [ ! "${LINKER}" ]; then
  LINKER=lld
fi
check_program ${LINKER}

PROJECT_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH:-.}
if [ ! "${PROJECT_DIR}" ]; then
  echo "PROJECT_DIR source path not set"
  exit 1
fi
if [ ! "${BUILD_DIR}" ]; then
  BUILD_DIR="/tmp/tpp-llvm"
fi
BUILD_DIR=$(realpath ${BUILD_DIR})
BUILD_DIR=${BUILD_DIR:-build-${COMPILER}}
mkdir -p ${BUILD_DIR}

 # Configure LLVM
echo "--- CONFIGURE"
echo_run cmake -Wno-dev -G Ninja ${PROJECT_DIR}/llvm \
    -B${BUILD_DIR} -S${PROJECT_DIR} \
    -DLLVM_ENABLE_PROJECTS=${LLVM_PROJECTS} \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD=${LLVM_TARGETS} \
    -DCMAKE_BUILD_TYPE=${KIND} \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DLLVM_USE_LINKER=${LINKER} \
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}

# Build LLVM
echo "--- BUILD"
echo_run ninja -C ${BUILD_DIR} all
if [ $? != 0 ]; then
  exit 1
fi

# Check LLVM
if [ "1" == "${CHECK}" ]; then
  echo "--- CHECK"
  echo_run ninja -C ${BUILD_DIR} check-all
  if [ $? != 0 ]; then
    exit 1
  fi
fi

 # Install LLVM
 echo "--- INSTALL"
 echo_run ninja -C ${BUILD_DIR} install
if [ $? != 0 ]; then
  exit 1
fi
