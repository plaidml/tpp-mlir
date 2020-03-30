# An out-of-tree dialect template for MLIR

This repository contains a template for an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a
standalone `opt`-like tool to operate on that dialect.

## How to build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mdir build && cd build
cmake -G Ninja .. -DCMAKE_LINKER=lld -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone
```
To build the documentation from the TableGen description of the dialect
operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

----------

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)]("http://creativecommons.org/publicdomain/zero/1.0/")

To the extent possible under law,
[Jean-Michel Gorius](https://jmgorius.gitlab.io/) has waived all copyright and
related or neighboring rights to this MLIR dialect template.
