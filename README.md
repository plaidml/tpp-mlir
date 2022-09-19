# TPP Sandbox

This is an experiment in using MLIR to automatically select the best [Tensor Processing Primitives](https://arxiv.org/abs/2104.05755) for ML models.

This repository contains an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a standalone `opt`-like tool to operate on that dialect.

It also contains the recipes to use [LIBXSMM](https://github.com/libxsmm/libxsmm) from inside MLIR and can be used by other tools to drive our passes.

There's [work in progress](https://github.com/iree-org/iree/tree/tpp) inside [IREE](https://iree-org.github.io/iree/) to use this sandbox's work on their pipeline.

## Build Status

[![Standalone build status](https://badge.buildkite.com/7c04eb392db7ba16b30684d80e0e4320254f7cf61558c6336f.svg)](https://buildkite.com/intel/tpp-compiler)

## How to build LLVM

```sh
# Clone
https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
pushd llvm-project/build

# This is important for the next step
export CUSTOM_LLVM_ROOT=`pwd`
echo $CUSTOM_LLVM_ROOT
export PATH=$CUSTOM_LLVM_ROOT/bin:$PATH

# Build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_USE_LINKER=lld
ninja 

popd
```

## How to build

This setup assumes that you have built LLVM and MLIR in `$CUSTOM_LLVM_ROOT` as above.

```sh
# Clone
git clone https://github.com/plaidml/tpp-sandbox.git
mkdir tpp-sandbox/build
pushd tpp-sandbox/build

# Build & test
cmake -G Ninja .. \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DMLIR_DIR=$CUSTOM_LLVM_ROOT/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$CUSTOM_LLVM_ROOT/bin/llvm-lit
cmake --build . --target check-standalone-opt

popd
```

To build the documentation from the TableGen description of the dialect
operations, run:

```sh
cmake --build . --target mlir-doc
```

## License

This dialect template is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.

## TODO:

1. Align runtime with IREE see: https://github.com/iree-org/iree/tree/main/compiler/src/iree/compiler/Dialect/VMVX (You don't want to pass the memref to the runtime but the single pointer + offset + strides).

## Note:

- https://libxsmm.readthedocs.io/en/latest/libxsmm_aux/#meta-image-file-io
- Nice link for conv: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html

- in IREE: Codegen/Common/ConvertToDestinationPassingStylePass.cpp

- Set up bufferization options: https://github.com/llvm/llvm-project/commit/c66303c2870c9a77a0f2a8aa16fd0ea87b0358e6

- https://reviews.llvm.org/D129217 See 'getMixedSizes'
