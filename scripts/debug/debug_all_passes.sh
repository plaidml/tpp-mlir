#!/usr/bin/env bash

SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

TMP_DIR=$(mktemp -d)
DUMP_FILE=${TMP_DIR}/dump.mlir
SPLIT=${SCRIPT_DIR}/debug/split.py
DIFF=${SCRIPT_DIR}/debug/diff.py

ROOT_DIR=$(git_root)
DIFF_TOOL=diff
BIN_DIR=$ROOT_DIR/build/bin
while getopts "b:d:m:o:" arg; do
  case ${arg} in
    b)
      BIN_DIR=$(realpath ${OPTARG})
      if [ ! -x ${BIN_DIR}/mlir-gen ]; then
        echo "'${OPTARG}' not a bin directory"
        exit 1
      fi
      ;;
    d)
      DIFF_TOOL=${OPTARG}
      check_program ${DIFF_TOOL}
      ;;
    m)
      MLIR_GEN_FLAGS=${OPTARG}
      ;;
    o)
      TPP_OPT_FLAGS=${OPTARG}
      ;;
    *)
      echo "Invalid option: ${OPTARG}"
      exit 1
  esac
done
MLIR_GEN=${BIN_DIR}/mlir-gen
TPP_OPT=${BIN_DIR}/tpp-opt

## Get IR dump
echo "Producing dump at ${TMP_DIR}"
${MLIR_GEN} --bias --relu ${MLIR_GEN_FLAGS} | \
    ${TPP_OPT} \
      ${TPP_OPT_FLAGS} \
      --default-tpp-passes \
      --mlir-print-ir-after-all \
      > /dev/null 2> ${DUMP_FILE}

## Split dump
echo "Splitting the file into multiple outputs"
pushd ${TMP_DIR}
${SPLIT} ${DUMP_FILE}
# Quick idea of how many files
ls -l ${TMP_DIR}/???.mlir | head -n 3
echo "..."
ls -l ${TMP_DIR}/???.mlir | tail -n 3

## Diff the stages
echo "Diffing the files with ${DIFF_TOOL}"
${DIFF} -d ${DIFF_TOOL} mlir
popd
