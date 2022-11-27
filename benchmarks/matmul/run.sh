#!/bin/bash

source ../common.sh

run_benchmark -p matmul -i 10 12x6x9 $*
run_benchmark -p matmul -i 10 48x64x96 $*
run_benchmark -p matmul -i 10 64x48x96 $*
run_benchmark -p matmul -i 10 64x64x64 $*
