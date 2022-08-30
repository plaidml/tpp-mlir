#!/bin/bash

source ../common.sh

compile () {
  echo "Compile driver ----> matmul_driver_${1}"
  echo "Compile kernel ----> matmul_kernel_${1}"
  
  # Compile driver. 
  clang -O3 -emit-llvm -S -I $LIB_INCLUDE_PATH -DARG_MNK=\"${1}\" matmul_driver.c
  llc matmul_driver.ll

  # Fire tpp compiler (with xsmm conversion).
  standalone-opt matmul_kernel_${1}.mlir -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize

  standalone-opt matmul_kernel_${1}.mlir -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize \
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

  clang -O3 matmul_driver.s matmul_kernel_${1}.s -L$LIB_PATH -lstandalone_c_runner_utils -lm -o matmul_${1}

  rm *.s
  rm *.ll
}

execute () {
  # Execute and check result.
  if ./matmul_${1} >matmul_${1}.log 2>&1; then
    grep "LIBXSMM: ..* GFLOPS\/s" matmul_${1}.log
    printf "${GREEN} OK ${NC} \n"
  else
    printf "${RED} Oh NO ${NC} \n";
    exit 1
  fi 
  
  #rm matmul_${1}.log
  #rm matmul_${1}
}


echo "--- MATMUL 12x6x9"
compile "12x6x9"
execute "12x6x9"

echo "--- MATMUL 64x48x96"
compile "64x48x96"
execute "64x48x96"

echo "--- MATMUL 48x64x96"
compile "48x64x96"
execute "48x64x96"

echo "--- MATMUL 64x64x64"
compile "64x64x64"
execute "64x64x64"
