#!/usr/bin/env bash
#
# Common functions to all scripts
# Usage: source common.sh

# Find the git root directory
git_root() {
  git rev-parse --show-toplevel
}

# Check if a program is in the PATH
check_program() {
  PROG=$1
  if ! which $PROG > /dev/null; then
    echo "Required program '$PROG' not found"
    exit 1
  fi
}

# Echoes and runs a program
echo_run() {
  PROGRAM=$*
  echo ${PROGRAM}
  ${PROGRAM}
}

# Get the LLVM version for this build
llvm_version() {
  LLVM_VERSION_FILE=$(git_root)/build_tools/llvm_version.txt
  if [ ! -f "${LLVM_VERSION_FILE}" ]; then
    echo "Cannot find LLVM version file in repo $PWD"
    exit 1
  fi
  LLVM_VERSION=$(cat ${LLVM_VERSION_FILE})
  if [ ! "${LLVM_VERSION}" ]; then
    echo "Cannot find LLVM version on current repository"
    exit 1
  fi
  echo ${LLVM_VERSION}
}
