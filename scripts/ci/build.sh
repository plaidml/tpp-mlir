#!/usr/bin/env bash
#
# Builds the project (tests, installs, post-install)

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

die_syntax() {
  echo "Syntax: $0 -b BLD_DIR [-c] [-i] [-p] [-B]"
  echo ""
  echo "  -c: Optional, runs check-all"
  echo "  -i: Optional, installs"
  echo "  -p: Optional, post-install (copy all build stuff to install dir)"
  echo "  -B: Optional, runs benchmarks"
  exit 1
}

# Cmd-line opts
TARGETS="all"
while getopts "b:cipB" arg; do
  case ${arg} in
    b)
      BLD_DIR=$(realpath ${OPTARG})
      if [ ! -f ${BLD_DIR}/CMakeCache.txt ]; then
        echo "'${OPTARG}' not a build directory"
        die_syntax
      fi
      ;;
    c)
      TARGETS="${TARGETS} check-all"
      ;;
    i)
      TARGETS="${TARGETS} install"
      ;;
    p)
      POST_INSTALL=1
      ;;
    B)
      TARGETS="${TARGETS} benchmarks"
      ;;
    ?)
      echo "Invalid option: ${OPTARG}"
      die_syntax
      ;;
  esac
done

# Mandatory arguments
if [ ! "${BLD_DIR}" ]; then
  die_syntax
fi

# Build all requested targets
echo_run ninja -C ${BLD_DIR} ${TARGETS}
if [ $? != 0 ]; then
  exit 1
fi

# Post Install (is this really needed?)
if [ "${POST_INSTALL}" != "" ]; then
  CMAKE_CACHE_FILE=${BLD_DIR}/CMakeCache.txt
  INST_DIR=$(grep "CMAKE_INSTALL_PREFIX:PATH=" ${CMAKE_CACHE_FILE} | cut -d"=" -f2)
  if not cp -rv ${BLD_DIR}/* ${INST_DIR}; then
    echo "Error on post-install"
    exit 1
  fi
fi
