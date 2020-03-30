# An out-of-tree dialect template for MLIR

This repository contains a template for an out-of-tree MLIR dialect as well as a
standalone `opt`-like tool to operate on that dialect.

## How to build

This setup assumes that you have built LLVM in `$BUILD_DIR` and installed it to `$PREFIX`. To build and launch the tests, run
```sh
mdir build && cd build
cmake -G Ninja .. -DCMAKE_LINKER=lld -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone
```
