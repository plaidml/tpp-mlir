#!/bin/bash

source ../common.sh

DRIVER=simple_copy_driver
KERNEL=simple_copy_kernel

# Compile driver. 
clang -O3 -emit-llvm -S ${DRIVER}.c
llc ${DRIVER}.ll

# Fire tpp compiler.
standalone-opt ${KERNEL}.mlir -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp

standalone-opt ${KERNEL}.mlir -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-linalg-to-loops -arith-expand -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts | mlir-translate -mlir-to-llvmir -o ${KERNEL}.ll
llc ${KERNEL}.ll

# Merge them.
unamestr=$(uname)
if [[ "$unamestr" == 'Darwin' ]]; then
  export DYLD_LIBRARY_PATH=$LIB_PATH
else
  export LD_LIBRARY_PATH=$LIB_PATH
fi
clang -O3 ${DRIVER}.s ${KERNEL}.s -L$LIB_PATH -lstandalone_c_runner_utils -o copy

# Execute and check result.
./copy > result.txt 2>&1

if cat result.txt | grep "Result is correct" &> /dev/null ; then
  printf "${GREEN} OK ${NC} \n"
else
  printf "${RED} Oh NO ${NC} \n";
fi

rm copy
rm *.s
rm *.ll
rm result.txt
