#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# Check LLVM installation and trigger the build if not.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

LLVMROOT=${HOME}/installs/llvm
if [ ! -d ${LLVMROOT} ]; then
  mkdir -p ${LLVMROOT}
fi

# Find LLVM_VERSION
LLVM_VERSION=$(llvm_version)

# If not found, trigger a build
if [ ! -d "${LLVMROOT}/${LLVM_VERSION}" ]; then
  ${HOME}/scripts/trigger.sh tpp-llvm ${LLVM_VERSION}
else
  echo "Found $LLVM_VERSION"
fi
