#!/bin/bash

source ../common.sh

compile () {
  echo "Compile driver ----> mlp_driver"
  echo "Compile kernel ----> mlp_kernel"
 
  clang -O3 -emit-llvm -S -I$LIB_INCLUDE_PATH -DARG_MNK=\"${1}\" mlp_driver.c
  llc $LLC_ARGS mlp_driver.ll

  tpp-opt mlp_kernel.mlir -map-linalg-to-tpp -main-closure -pre-bufferization -loop-invariant-code-motion -canonicalize -undo-main-closure -tile-consumer-and-fuse-producers="tile-sizes=1,0,0,0" -canonicalize -tile-consumer-and-fuse-producers="tile-sizes=1,0,0" -canonicalize -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize -map-linalg-to-tpp -convert-linalg-to-tpp="use-parallel-loops=false" -map-to-brgemm -convert-linalg-to-tpp -convert-tpp-to-xsmm 


  # Let's avoid calling the sparse compiler to lower to LLVM.
  tpp-opt mlp_kernel.mlir -map-linalg-to-tpp -main-closure -pre-bufferization -loop-invariant-code-motion -canonicalize -undo-main-closure -tile-consumer-and-fuse-producers="tile-sizes=1,0,0,0" -canonicalize -tile-consumer-and-fuse-producers="tile-sizes=1,0,0" -canonicalize -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize -map-linalg-to-tpp -convert-linalg-to-tpp="use-parallel-loops=false" -map-to-brgemm -convert-linalg-to-tpp -convert-tpp-to-xsmm -loop-invariant-code-motion -convert-xsmm-to-func -convert-linalg-to-loops -arith-expand -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -sparse-compiler | mlir-translate -mlir-to-llvmir -o mlp_kernel.ll

  llc $LLC_ARGS mlp_kernel.ll


  # Merge them.
  unamestr=$(uname)
  if [[ "$unamestr" == 'Darwin' ]]; then
    export DYLD_LIBRARY_PATH=$LIB_PATH:$DYLD_LIBRARY_PATH
  else
    export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH
  fi

  clang -O3 mlp_driver.s mlp_kernel.s -L$LIB_PATH -ltpp_c_runner_utils -lm -o mlp
 
  rm *.s
  rm *.ll
}

execute () {
  cat /dev/null >mlp.log

  # Execute and check result based on MLIR toolchain.
  if [ -e ./mlp ] && ./mlp >>mlp.log 2>&1; then
    #grep "MLIR: ..* GFLOPS\/s" mlp_${1}.log
    # Execute tpp mlp driver.
    #if [ -e ./mlp ] && ./mlp 0 ${1} >>mlp_${1}.log 2>&1; then
    #  grep "XSMM: ..* GFLOPS\/s" mlp_${1}.log
    #fi
    printf "${GREEN} OK ${NC} \n"
  else
    printf "${RED} Oh NO ${NC} \n";
    exit 1
  fi 
}

# Compile tpp mlp driver without MLIR toolchain.
#clang -O3 mlp_driver.c -I$LIB_INCLUDE_PATH -L$LIB_PATH -ltpp_c_runner_utils -lm -o mlp

# Compile and execute kernels related to MLIR files.
for MLIR in ./mlp_kernel_*.mlir; do
  KERNEL=$(echo "${MLIR}" | xargs -I{} basename {} .mlir | cut -d_ -f3)
  echo "--- MLP ${KERNEL}"
  compile "${KERNEL}"
  execute "${KERNEL}"
done
