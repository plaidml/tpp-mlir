#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# Check LLVM installation.
# Optionally, trigger a build if LLVM is not found.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

LLVMROOT=${HOME}/installs/llvm
mkdir -p ${LLVMROOT}

# Find LLVM_VERSION
LLVM_VERSION=$(llvm_version)

if [ -d "${LLVMROOT}/${LLVM_VERSION}" ]; then
  echo "Found $LLVM_VERSION"
  exit 0
fi

# LLVM not found, trigger a build if requested.
if [ "1" == "${BUILD}" ]; then
  COMMIT_SHA=$(git_commit)
  ${SCRIPT_DIR}/ci/trigger.sh -p tpp-llvm -c ${COMMIT_SHA}
fi
