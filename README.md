# An out-of-tree dialect template for MLIR

This repository contains a template for an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a
standalone `opt`-like tool to operate on that dialect.

## Build Status

[![Build status](https://badge.buildkite.com/7c04eb392db7ba16b30684d80e0e4320254f7cf61558c6336f.svg)](https://buildkite.com/intel/tpp-compiler)

## How to build LLVM

```
# creating project dir
mkdir -p tpp_compiler_sandbox/
cd tpp_compiler_sandbox/

git clone -b sandbox https://github.com/chelini/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
export CUSTOM_LLVM_ROOT=`pwd`
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON 

ninja 
echo $CUSTOM_LLVM_ROOT
export PATH=$CUSTOM_LLVM_ROOT/bin:$PATH
cd ../..

# clone and building the TPP compiler Sandbox
git clone https://github.com/plaidml/tpp-sandbox.git
mkdir tpp-sandbox/build
cd tpp-sandbox/build
cmake -G Ninja .. -DMLIR_DIR=$CUSTOM_LLVM_ROOT/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$CUSTOM_LLVM_ROOT/bin/llvm-lit
cmake --build . --target check-standalone-opt

```

## How to build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone-opt
```
To build the documentation from the TableGen description of the dialect
operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

## License

This dialect template is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.

## TODO:

1. Align runtime with IREE see: https://github.com/iree-org/iree/tree/main/compiler/src/iree/compiler/Dialect/VMVX (You don't want to pass the memref to the runtime but the single pointer + offset + strides).

## Note:

- check: getSubMap
- check: getSliceMap
- https://libxsmm.readthedocs.io/en/latest/libxsmm_aux/#meta-image-file-io
- Nice link for conv: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html

```
#trait1 = {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,  // input a
    affine_map<(i, j) -> (i, j)>   // input b
    affine_map<(i, j) -> (i, j)>   // output c
  ],
  iterator_types = ["parallel", "parallel"],
}

tpp.add ins(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) out(tensor<2x2xf32) #trait1
```

How to match something like:

```
func.func @add_d(%arga: tensor<32xf32, #DV>, %argb: f32, %argx: tensor<32xf32>) -> tensor<32xf32> {
  %0 = linalg.generic #trait1
     ins(%arga: tensor<32xf32, #DV>)
    outs(%argx: tensor<32xf32>) {
      ^bb(%a: f32, %x: f32):
        %0 = arith.addf %a, %argb : f32
        linalg.yield %0 : f32
  } -> tensor<32xf32>
  return %0 : tensor<32xf32>
}
```

- in IREE: Codegen/Common/ConvertToDestinationPassingStylePass.cpp

- Set up bufferization options: https://github.com/llvm/llvm-project/commit/c66303c2870c9a77a0f2a8aa16fd0ea87b0358e6

- https://reviews.llvm.org/D129217 See 'getMixedSizes'
