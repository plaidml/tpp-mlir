#!/usr/bin/env bash
# shellcheck disable=SC1091
#
# Setup runtime environment based on build_tools/llvm_version.txt

if [ ! "${TPPROOT}" ] && [ -d /nfs_home/buildkite-slurm/builds/tpp ]; then
  source /nfs_home/buildkite-slurm/builds/tpp/enable-tpp
fi

if [ "${TPP_LLVM}" ]; then
  # basic utilities functions (git_root, llvm_version)
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/ci/common.sh"

  # LLVM version used to build TPP-mlir
  if [ ! "${TPP_LLVM_VERSION}" ]; then
    TPP_LLVM_VERSION=$(llvm_version)
    if [ "${TPP_LLVM_VERSION}" ]; then
      export TPP_LLVM_VERSION;
    fi
  fi

  if [ "${TPP_LLVM_VERSION}" ]; then
    # setup environment
    export TPP_LLVM_DIR=${TPP_LLVM}/${TPP_LLVM_VERSION}
    # avoid overriding PATH/LD_LIBRARY_PATH of initial environment (append)
    export PATH=${PATH}:${TPP_LLVM_DIR}/bin
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${TPP_LLVM_DIR}/lib

    # setup additional/legacy envronment variables
    export CUSTOM_LLVM_ROOT=${TPP_LLVM_DIR}
    export LLVM_VERSION=${TPP_LLVM_VERSION}
  else
    echo "ERROR: Cannot determine LLVM-version!"
  fi
else
  echo "ERROR: Please source the TPP-environment first!"
fi
