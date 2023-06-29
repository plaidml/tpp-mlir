#!/usr/bin/env bash
#
# Runs CMake on the source / build directories

# Include common utils
source $(realpath $(dirname $0))/common.sh

die_syntax() {
  echo "Syntax: $0 -s SRC_DIR -b BLD_DIR -m MLIR_DIR [-i INST_DIR]"
  echo "          [-t (Release|Debug|RelWithDebInfo)] [-c (clang|gcc)] [-g (gcc toolchain)]"
  echo "          [-l (ld|lld|gold|mold)] [-R] [-S] [-n N]"
  echo ""
  echo "  -i: Optional install dir, defaults to system"
  echo "  -t: Optional build type flag, defaults to Release"
  echo "  -c: Optional compiler flag, defaults to clang"
  echo "  -g: Optional gcc toolchain flag, may be needed by clang"
  echo "  -l: Optional linker flag, defaults to system linker"
  echo "  -R: Optional request to remove BLD_DIR before CMake"
  echo "  -S: Optional sanitizer flag, defaults to none"
  echo "  -G: Optional GPU support flag, defaults to none"
  echo "  -n: Optional link job flag, defaults to nproc"
  exit 1
}

# Cmd-line opts
while test $# -gt 0; do
  case "$1" in
    -s)
      SRC_DIR=$(realpath $2)
      if [ ! -d ${SRC_DIR}/.git ]; then
        echo "Source '$2' not a Git directory"
        die_syntax
      fi
      shift 2;;
    -b)
      BLD_DIR=$(realpath $2)
      if ! mkdir -p ${BLD_DIR}; then
        echo "Error creating build directory '$2'"
        die_syntax
      fi
      shift 2;;
    -i)
      INST_DIR=$(realpath $2)
      if ! mkdir -p ${INST_DIR}; then
        echo "Error creating install directory '$2'"
        die_syntax
      fi
      shift 2;;
    -m)
      MLIR_DIR=$(realpath $2)
      if [ ! -f ${MLIR_DIR}/MLIRConfig.cmake ]; then
        echo "MLIR '$2' not a CMake directory"
        die_syntax
      fi
      shift 2;;
    -g)
      GCC_DIR=$(realpath $2)
      GCC_TOOLCHAIN_OPTIONS="-DCMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN=${GCC_DIR} -DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=${GCC_DIR}"
      shift 2;;
    -c)
      if [ "$2" == "clang" ]; then
        check_program clang
        check_program clang++
        CC=clang
        CXX=clang++
      elif [ "$2" == "gcc" ]; then
        check_program gcc
        check_program g++
        CC=gcc
        CXX=g++
      else
        echo "Compiler "$2" not recognized"
        die_syntax
      fi
      shift 2;;
    -t)
      if [ "$2" == "Release" ]; then
        BUILD_OPTIONS="${BUILD_OPTIONS} -DCMAKE_BUILD_TYPE=$2"
      elif [ "$2" == "Debug" ]; then
        BUILD_OPTIONS="${BUILD_OPTIONS} -DCMAKE_BUILD_TYPE=$2"
      elif [ "$2" == "RelWithDebInfo" ]; then
        BUILD_OPTIONS="${BUILD_OPTIONS} -DCMAKE_BUILD_TYPE=$2"
      else
        echo "Build type "$2" not recognized"
        die_syntax
      fi
      shift 2;;
    -l)
      if [ "$2" == "ld" ]; then
        check_program ld
      elif [ "$2" == "lld" ]; then
        check_program lld
      elif [ "$2" == "gold" ]; then
        check_program gold
      elif [ "$2" == "mold" ]; then
        check_program mold
      else
        echo "Linker "$2" not recognized"
        die_syntax
      fi
      LINKER_OPTIONS="${LINKER_OPTIONS} -DLLVM_USE_LINKER=$2"
      shift 2;;
    -R)
      REMOVE_BLD_DIR=1
      shift;;
    -S)
      SAN_OPTIONS="-DUSE_SANITIZER=\"Address;Memory;Leak;Undefined\""
      shift;;
    -G)
      if [ "$2" ] && [[ "$2" != "-"* ]]; then
        ENABLE_GPU="-DTPP_GPU=$2"
        shift 2
      else  # legacy
        ENABLE_GPU="-DTPP_GPU=cuda"
        shift
      fi
      ;;
    -n)
      PROCS=$(nproc)
      if [ "$2" -gt "0" ] && [ "$2" -lt "${PROCS}" ]; then
        LINKER_OPTIONS="${LINKER_OPTIONS} -DCMAKE_JOB_POOL_LINK=link -DCMAKE_JOB_POOLS=link=$2"
      else
        echo "Invalid value for number of linker jobs '$2'"
        die_syntax
      fi
      shift 2;;
    *)
      echo "Invalid option: $2"
      die_syntax
      shift;;
  esac
done

# Mandatory arguments
if [ ! "${BLD_DIR}" ] || [ ! "${SRC_DIR}" ] || [ ! "${MLIR_DIR}" ]; then
  die_syntax
fi

if [ ! "${BUILD_OPTIONS}" ]; then
  BUILD_OPTIONS="-DCMAKE_BUILD_TYPE=Release"
fi

if [ ! "${CC}" ] || [ ! "${CXX}" ]; then
  check_program clang
  check_program clang++
  CC=clang
  CXX=clang++
fi

# Check deps
check_program cmake
check_program ninja
check_program pip
pip install --upgrade --user lit
check_program lit
pip install --upgrade --user -r ${SRC_DIR}/benchmarks/harness/requirements.txt

TPP_LIT=$(which lit)
# patch incorrect interpreter
if [ "${TPP_LIT}" ] && [ "$(command -v sed)" ]; then
  sed -i 's/#!\/usr\/bin\/python3/#!\/usr\/bin\/env python3/' ${TPP_LIT}
fi

# Consider to remove BLD_DIR shortly before running CMake
if [ "${REMOVE_BLD_DIR}" ] && [ "0" != "${REMOVE_BLD_DIR}" ]; then
  echo_run rm -rf ${BLD_DIR}
fi

# CXX: simple check for external toolchain argument
read -ra CMAKE_CXX <<<"${CXX}"
if [[ "${CMAKE_CXX[@]:1}" == "--gcc-toolchain="* ]]; then
  GCC_TOOLCHAIN_OPTIONS+=" -DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=$(cut -d= -f2 <<<"${CMAKE_CXX[@]:1}")"
  CXX=${CMAKE_CXX[0]}
fi
# CC: simple check for external toolchain argument
read -ra CMAKE_CC <<<"${CC}"
if [[ "${CMAKE_CC[@]:1}" == "--gcc-toolchain="* ]]; then
  GCC_TOOLCHAIN_OPTIONS+=" -DCMAKE_CC_COMPILER_EXTERNAL_TOOLCHAIN=$(cut -d= -f2 <<<"${CMAKE_CC[@]:1}")"
  CC=${CMAKE_CC[0]}
fi

# CMake
echo_run cmake -Wno-dev -G Ninja \
    -B${BLD_DIR} -S${SRC_DIR} \
    -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_INSTALL_PREFIX=${INST_DIR} \
    -DMLIR_DIR=${MLIR_DIR} \
    -DLLVM_EXTERNAL_LIT=${TPP_LIT} \
    ${BUILD_OPTIONS} \
    ${GCC_TOOLCHAIN_OPTIONS} \
    ${LINKER_OPTIONS} \
    ${SAN_OPTIONS} \
    ${ENABLE_GPU}
