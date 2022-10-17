#!/bin/bash

HERE=$(cd "$(dirname "$0")" && pwd -P)
source ${HERE}/../common.sh

compile () {
  echo "Compile driver ----> matmul_driver_${1}"
  echo "Compile kernel ----> matmul_kernel_${1}"
  
  # Compile driver. 
  clang -O3 -emit-llvm -S -I$LIB_INCLUDE_PATH -DARG_MNK=\"${1}\" matmul_driver.c
  llc matmul_driver.ll

  # Fire tpp compiler (with xsmm conversion).
  tpp-opt matmul_kernel_${1}.mlir -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize

  tpp-opt matmul_kernel_${1}.mlir -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize \
    -convert-linalg-to-tpp="enable-tiling" -convert-tpp-to-xsmm -loop-invariant-code-motion -convert-xsmm-to-func \
    -convert-linalg-to-loops -arith-expand -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm \
    -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts \
  | mlir-translate -mlir-to-llvmir -o matmul_kernel_${1}.ll
  llc matmul_kernel_${1}.ll

  # Merge them.
  unamestr=$(uname)
  if [[ "$unamestr" == 'Darwin' ]]; then
    export DYLD_LIBRARY_PATH=$LIB_PATH:$DYLD_LIBRARY_PATH
  else
    export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH
  fi

  clang -O3 matmul_driver.s matmul_kernel_${1}.s -L$LIB_PATH -ltpp_c_runner_utils -lm -o matmul_${1}

  rm *.s
  rm *.ll
}

execute () {
  cat /dev/null >matmul_${1}.log

  # Execute and check result based on MLIR toolchain.
  if [ -e ./matmul_${1} ] && ./matmul_${1} >>matmul_${1}.log 2>&1; then
    grep "MLIR: ..* GFLOPS\/s" matmul_${1}.log
    # Execute TPP matmul driver.
    if [ -e ./matmul ] && ./matmul 0 ${1} >>matmul_${1}.log 2>&1; then
      grep "XSMM: ..* GFLOPS\/s" matmul_${1}.log
    fi
    printf "${GREEN} OK ${NC} \n"
  else
    printf "${RED} Oh NO ${NC} \n";
    exit 1
  fi 
}

# Compile TPP matmul driver without MLIR toolchain.
clang -O3 matmul_driver.c matmul_kernel.c -I$LIB_INCLUDE_PATH -L$LIB_PATH -ltpp_c_runner_utils -lm -o matmul

# Compile and execute kernels related to MLIR files.
for MLIR in ./matmul_kernel_*.mlir; do
  KERNEL=$(echo "${MLIR}" | xargs -I{} basename {} .mlir | cut -d_ -f3)
  echo "--- MATMUL ${KERNEL}"
  compile "${KERNEL}"
  execute "${KERNEL}"
done
