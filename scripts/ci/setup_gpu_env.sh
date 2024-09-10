#!/usr/bin/env bash
#
# Sets up GPU environment.
# Usage: source setup_gpu_env.sh

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

# Env CUDA setup
if [[ ${GPU,,} =~ "cuda" ]]; then
  echo "Setting up CUDA environment"
  echo "Hard-coding CUDA-compatible GCC version (12.3)"
  source /swtools/gcc/gcc-12.3.0/gcc_vars.sh
  source /swtools/cuda/latest/cuda_vars.sh
  check_program nvcc
fi
