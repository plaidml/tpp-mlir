#!/usr/bin/env bash
#
# Common functions to all scripts
# Usage: source common.sh

# Find the git root directory
git_root() {
  if [ "$(command -v git)" ]; then
    git rev-parse --show-toplevel
  else
    echo "ERROR: missing prerequisites!"
    exit 1
  fi
}

# Find the current git commit SHA
git_commit() {
  if [ "$(command -v git)" ]; then
    git rev-parse HEAD
  else
    echo "ERROR: missing prerequisites!"
    exit 1
  fi
}

# Check if a program is in the PATH
check_program() {
  PROG=$1
  if ! which $PROG > /dev/null; then
    echo "ERROR: '$PROG' not found!"
    exit 1
  fi
}

# Echoes and runs a program
echo_run() {
  PROGRAM=$*
  echo "${PROGRAM}"
  ${PROGRAM}
}

# Get the LLVM version for this build
llvm_version() {
  LLVM_VERSION_FILE=$(git_root)/build_tools/llvm_version.txt
  if [ ! -f "${LLVM_VERSION_FILE}" ]; then
    echo "ERROR: cannot find ${LLVM_VERSION_FILE} for ${PWD}!"
    exit 1
  fi
  LLVM_VERSION=$(cat "${LLVM_VERSION_FILE}")
  if [ ! "${LLVM_VERSION}" ]; then
    echo "ERROR: cannot find LLVM version in ${LLVM_VERSION_FILE}!"
    exit 1
  fi
  echo "${LLVM_VERSION}"
}
