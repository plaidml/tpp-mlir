#!/usr/bin/env bash
#
# Checks code formatting

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

CLANG_FORMAT=$(which clang-format-16)
if [ ! "${CLANG_FORMAT}" ]; then
  echo "This script needs clang-format-16 to work"
  echo "Please install the tool and run again"
  exit 1
fi

# If -i is passed, actually change the formatting in place
# This should NEVER be used by CI, but by developers trying
# to conform to the CI checks.
if [ "$1" == "-i" ]; then
  INPLACE=1
fi

ERRORS=0

_clang_format() {
  DIR="$1"
  PATTERN="$2"
  FILES=$(find "${DIR}" -type f -name "${PATTERN}")
  for FILE in ${FILES}; do
    OUT=$(${CLANG_FORMAT} -n ${FILE} 2>&1)
    if [ "${OUT}" ]; then
      if [ $INPLACE ]; then
        ${CLANG_FORMAT} -i ${FILE}
        echo "File ${FILE} updated"
      fi
      ERRORS=$((ERRORS+1))
      echo "${OUT}"
    fi
  done
}

clang_format() {
  DIR="$1"
  _clang_format ${DIR} "*.cpp"
  _clang_format ${DIR} "*.h"
}

ROOT=$(git_root)
clang_format "${ROOT}/lib"
clang_format "${ROOT}/include"
clang_format "${ROOT}/runtime"
clang_format "${ROOT}/tools"

# Returning zero always means there were no changes
# Returns non-zero to break CI if there are.
if [ ${ERRORS} -gt 0 ]; then
  echo "${ERRORS} files contain formatting errors"
  exit 1
fi
