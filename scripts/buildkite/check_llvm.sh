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
mkdir -p ${LLVMROOT}

# Find LLVM_VERSION
LLVM_VERSION=$(llvm_version)

# If not found, trigger a build
if [ ! -d "${LLVMROOT}/${LLVM_VERSION}" ]; then
  COMMIT_SHA=$(git_commit)
  ${SCRIPT_DIR}/ci/trigger.sh tpp-llvm ${COMMIT_SHA}
else
  echo "Found $LLVM_VERSION"
fi
