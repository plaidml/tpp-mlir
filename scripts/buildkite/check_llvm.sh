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

# If not found, trigger a build
if [ "1" == "${BUILD}" ]; then
  COMMIT_SHA=$(git_commit)
  ${SCRIPT_DIR}/ci/trigger.sh -p tpp-llvm -c ${COMMIT_SHA}
else
  exit 1
fi
# if [ ! -d "${LLVMROOT}/${LLVM_VERSION}" ]; then
#   COMMIT_SHA=$(git_commit)
#   ${SCRIPT_DIR}/ci/trigger.sh tpp-llvm ${COMMIT_SHA}
# else
#   echo "Found $LLVM_VERSION"
# fi
