#!/bin/bash

source ../common.sh

compile () {
  echo "Compile driver ----> $1"
  echo "Compile kernel ----> $2"
  
  # Compile driver. 
  clang -O3 -emit-llvm -S -I $LIB_INCLUDE_PATH ${1}.c
  llc ${1}.ll

  # Fire tpp compiler (with xsmm conversion).
  standalone-opt ${2}.mlir -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp

  standalone-opt ${2}.mlir -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-linalg-to-loops -arith-expand -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts | mlir-translate -mlir-to-llvmir -o ${2}.ll
  llc ${2}.ll

  # Merge them.
  unamestr=$(uname)
  if [[ "$unamestr" == 'Darwin' ]]; then
    export DYLD_LIBRARY_PATH=$LIB_PATH
  else
    export LD_LIBRARY_PATH=$LIB_PATH
  fi

  clang -O3 ${1}.s ${2}.s -L$LIB_PATH -lstandalone_c_runner_utils -o matmul

  rm *.s
  rm *.ll
}

execute () {
  # Execute and check result.
  ./matmul > result.txt 2>&1
  rm matmul

 if cat result.txt | grep "Result is correct" &> /dev/null ; then
    printf "${GREEN} OK ${NC} \n"
  else
    printf "${RED} Oh NO ${NC} \n";
  fi 
  
  rm result.txt
}


# ----- matmul M = 12 N = 6 K = 9
compile "matmul_driver_12x6x9" "matmul_kernel_12x6x9"
execute

# --------- matmul M = 64 N = 48 and K = 96
compile "matmul_driver_64x48x96" "matmul_kernel_64x48x96"
execute

# --------- matmul M = 48 N = 64 and K = 96
compile "matmul_driver_48x64x96" "matmul_kernel_48x64x96"
execute
