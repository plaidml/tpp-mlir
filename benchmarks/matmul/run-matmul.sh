#!/bin/bash

# Reset
Color_Off='\033[0m'       # Text Reset

# Regular Colors
Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White

if ! command -v standalone-opt &> /dev/null
then
  echo "standalone-opt could not be found"
  exit
fi

if ! command -v mlir-translate &> /dev/null
then
  echo "mlir-translate could not be found"
  exit
fi

if ! command -v llc &> /dev/null
then
  echo "llc could not be found"
  exit
fi

if ! command -v clang &> /dev/null
then
  echo "clang could not be found"
  exit
fi

# Clang
which clang

# Assembler
which llc

# LLVM MLIR IR to LLVM IR
which mlir-translate

# TPP compiler
which standalone-opt

# Compiler driver 
clang -O3 -emit-llvm -S matmul_driver.c
llc matmul_driver.ll

# Fire tpp compiler
standalone-opt matmul_kernel.mlir -tpp-compiler | mlir-translate -mlir-to-llvmir -o matmul_kernel.ll
llc matmul_kernel.ll

# Merge them
clang -O3 matmul_driver.s matmul_kernel.s -o matmul

# Execute and check result
./matmul > result.txt 2>&1

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

if cat result.txt | grep "Result is correct" &> /dev/null ; then
  printf "${GREEN} OK ${NC} \n"
else
  printf "${RED} Oh NO ${NC} \n";
fi
