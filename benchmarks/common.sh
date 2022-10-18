#!/bin/bash

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

# Permit using $TPP_COMPILER (installed) rather than local build directory.
if [ ! "$TPP_COMPILER" ]; then
  if [ ! "$TPP_COMPILER_PREBUILT" ] || [ "0" = "$TPP_COMPILER_PREBUILT" ]; then
    TPP_COMPILER=$HERE/../build
  fi
fi

# This assumes the sandbox was built as described in the readme.
LIB_PATH=$TPP_COMPILER/lib
BIN_PATH=$TPP_COMPILER/bin

if [ "$LIBXSMMROOT" ]; then
  LIB_INCLUDE_PATH=$LIBXSMMROOT/include
else
  LIB_INCLUDE_PATH=$TPP_COMPILER/_deps/xsmm-src/include
fi

echo "lib path: ${LIB_PATH}"
echo "bin path: ${BIN_PATH}"
echo "lib include path: ${LIB_INCLUDE_PATH}"

# make tpp-opt (TPP compiler) available.
export PATH=${BIN_PATH}:$PATH

# LLVM options.
LLC_ARGS="-opaque-pointers --relocation-model=pic"
if [ "$LLVM_DIR" ]; then
  export PATH=$PATH:$LLVM_DIR/bin
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

if ! command -v tpp-opt &> /dev/null
then
  echo "tpp-opt could not be found"
  echo "Try: TPP_COMPILER=/path/to/tpp/build $0"
  exit
fi

if ! command -v mlir-translate &> /dev/null
then
  echo "mlir-translate could not be found"
  echo "Try: LLVM_DIR=/path/to/llvm/build $0"
  exit
fi

if ! command -v llc &> /dev/null
then
  echo "llc could not be found"
  echo "Try: LLVM_DIR=/path/to/llvm/build $0"
  exit
fi

if ! command -v clang &> /dev/null
then
  echo "clang could not be found"
  echo "If clang is built in, try:"
  echo "    LLVM_DIR=/path/to/llvm/build $0"
  echo "Otherwise, just install the clang package"
  exit
fi

# Clang.
echo "Using Clang in: "
which clang

# Assembler.
echo "Using Assembler in: "
which llc

# LLVM MLIR IR to LLVM IR.
echo "Using mlir translate in: "
which mlir-translate

# TPP compiler.
echo "Using tpp compiler in: "
which tpp-opt
