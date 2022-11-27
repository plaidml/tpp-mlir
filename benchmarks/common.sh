#!/bin/bash

# Output only when VERBOSE is set
function LOG() {
  if [ $VERBOSE ]; then
    echo "$*"
  fi
}

# The root dir is the git repository base directory
# This assumes you're running your script inside the TPP repository
ROOT_DIR=$(realpath $(dirname $0))
if git rev-parse --show-toplevel > /dev/null; then
  ROOT_DIR=$(git rev-parse --show-toplevel)
else
  echo "ERROR: Not running the script inside the TPP repository"
  exit 1
fi

# Find the build dir (if there's more than one, you must pass it as
# an argument to the script that includes this one via BUILD_DIR)
if [ -z "$BUILD_DIR" ]; then
  cmake_file=$(find "$ROOT_DIR" -name CMakeCache.txt | head -n 1)
  BUILD_DIR=$(realpath $(dirname "$cmake_file"))
fi

# Make sure we have a compiler available, prioritize user choices
CLANG=
if [ -d "$LLVM_DIR" ] && [ -x "$LLVM_DIR/bin/clang" ]; then
  CLANG="$LLVM_DIR/bin/clang"
elif command -v clang &> /dev/null; then
  CLANG=$(which clang)
else
  echo "clang could not be found"
  echo "If clang is built in, try:"
  echo "    LLVM_DIR=/path/to/llvm/build $*"
  echo "Otherwise, just install the clang package"
  exit 1
fi

LOG "Running parameters:"
LOG "    Compiler: $CLANG"
HARNESS="$ROOT_DIR/benchmarks/harness/controller.py"
LOG "     Harness: ${HARNESS}"
TEST_PATH="$ROOT_DIR/test/Benchmarks"
LOG "   TEST path: ${TEST_PATH}"

# Now we set all basic paths
LIB_PATH="$BUILD_DIR"/lib
if [ "$LIBXSMMROOT" ]; then
  LIB_INCLUDE_PATH="$LIBXSMMROOT"/include
else
  LIB_INCLUDE_PATH="$BUILD_DIR"/_deps/xsmm-src/include
fi
unamestr=$(uname)
if [[ "$unamestr" == 'Darwin' ]]; then
  export DYLD_LIBRARY_PATH=$LIB_PATH
else
  export LD_LIBRARY_PATH=$LIB_PATH
fi

LOG "    LIB path: ${LIB_PATH}"
LOG "LIB INC path: ${LIB_INCLUDE_PATH}"
LOG ""

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Common functions

# run_reference(source, num_it, *)
#
# - source: C source file, mandatory
# - num_it: number of iteratios, mandatory
# - *: extra arguments to the tool
#
# Note: A common extra arg is -x (for xsmm runs)
function run_reference() {
  SRC="$1.c"
  if [ ! -f "$SRC" ]; then
    >&2 echo "ERROR: File '$SRC' not found"
    exit 1
  fi
  DST="${SRC%.c}"
  ITER=$(($2+0))
  if [ "$ITER" == "0" ]; then
    >&2 echo "ERROR: Invalid iteration '$2'"
    exit 1
  fi
  shift 2
  ARGS="$*"
  CMD="$CLANG -O3 $SRC -L\"$LIB_INCLUDE_PATH\" -o $DST"
  if $CMD &> /dev/null; then
    # If it compiles, run
    CMD="./$DST -i $ITER $ARGS"
    if ! $CMD; then
      >&2 echo "ERROR: Running $DST"
      >&2 echo "   $CMD"
      exit 1
    fi
  else
    # Else, just bails
    >&2 echo "ERROR: Cannot compile '$SRC'"
    >&2 echo "   $CMD"
    exit 1
  fi
}

# run_tpp_mlir(source, num_it, alt, *)
#
# - source: MLIR source file, mandatory
# - num_it: number of iteratios, mandatory
# - alt: alternative name, if any (complement source), optional
# - *: extra arguments to the tool
#
# Note: alt name is "$source_$alt.mlir" (ex. matmul_12x24x48.mlir")
function run_tpp_mlir() {
  SRC="$TEST_PATH/$1.mlir"
  ITER=$(($2+0))
  if [ "$ITER" == "0" ]; then
    >&2 echo "ERROR: Invalid iteration '$2'"
    exit 1
  fi
  ALT_SRC="$TEST_PATH/$1_$3.mlir"
  if [ ! -f "$SRC" ]; then
    # Try with extension
    if [ -f "$ALT_SRC" ]; then
      SRC=$ALT_SRC
      shift
    else
      >&2 echo "ERROR: File '$SRC' not found"
      if [ -n "$3" ]; then
        >&2 echo "ERROR: File '$ALT_SRC' also not found"
      fi
      exit 1
    fi
  fi
  shift 2
  ARGS="$*"
  CMD="$HARNESS -q $SRC -n $ITER $ARGS"
  if ! $CMD; then
    >&2 echo "ERROR: Harness failed"
    >&2 echo "   $CMD"
    exit 1
  fi
}

# run_benchmark()
#
# - -p program: kernel name, mandatory
# - -i num_it: number of iterations, mandatory
# - -k: keep files (for debug), optional
# - *: arguments for the tools themselves (optional)
#
# Note: The arguments must be valid for both ref and mlir tools (ex: -v)
function run_benchmark() {
  TYPE="ref"
  ARGS=
  while getopts ":p:i:k" opt; do
    case "${opt}" in
      p)
        KERNEL=${OPTARG}
        ;;
      i)
        ITER=${OPTARG}
        ;;
      k)
        echo "Keeping binary for inspection"
        KEEP=1
        ;;
      *)
        # Append unrecognised args to extra
        ARGS="$ARGS -$OPTARG"
        ;;
    esac
  done
  shift $((OPTIND-1))
  ARGS="$ARGS $*"

  # Compile ninja code
  REFERENCE=$(run_reference "$KERNEL" "$ITER" $ARGS)
  echo "Reference output ($ARG): $REFERENCE"

  # Run the MLIR code
  OUTPUT=$(run_tpp_mlir "$KERNEL" "$ITER" $ARGS)
  echo " Compiler output ($ARG): $OUTPUT"

  # Cleanup
  if [ -z $KEEP ]; then
    rm -f "$KERNEL"
  fi
}
