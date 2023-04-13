#!/usr/bin/env bash
#
# Builds the project (tests, installs, post-install)

SCRIPT_DIR=$(realpath $(dirname $0))

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
      CHECK=1
      ;;
    i)
      INSTALL=1
      ;;
    p)
      POST_INSTALL=1
      ;;
    B)
      BENCHMARKS=1
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

# Build
if not ninja -C ${BLD_DIR}; then
  echo "Error building, will stop"
  exit 1
fi

# Test
if [ "${CHECK}" != "" ]; then
  if not ninja -C ${BLD_DIR} check-all; then
    echo "Error testing, will stop"
    exit 1
  fi
fi

# Install
if [ "${INSTALL}" != "" ]; then
  if not ninja -C ${BLD_DIR} install; then
    echo "Error installing, will stop"
    exit 1
  fi
fi

# Post Install
if [ "${POST_INSTALL}" != "" ]; then
  CMAKE_CACHE_FILE=${BLD_DIR}/CMakeCache.txt
  INST_DIR=$(grep "CMAKE_INSTALL_PREFIX:PATH=" ${CMAKE_CACHE_FILE} | cut -d"=" -f2)
  if not cp -rv ${BLD_DIR}/* ${INST_DIR}; then
    echo "Error on post-install"
    exit 1
  fi
fi

# Benchmarks
if [ "${BENCHMARKS}" != "" ]; then
  if not ninja -C ${BLD_DIR} benchmarks; then
    echo "Error running benchmarks"
    exit 1
  fi
fi
