# TPP MLIR Benchmarks

This directory contains benchmarks comparing MLIR passes of our compiler with manual implementaitons written in C++ calling libxsmm directly.
This allows us to measure how much off are we when compared with ninja-written code.

## Run Types

There are two types of runs: reference and XSMM, for each type of benchmark: MLIR compiler and ninja code.

### Reference Runs

These runs pure C++/MLIR without using TPP/XSMM, just executing the code as is, with compiler at -O3.
This is the most naive and inefficient way of running the kernels and lets us know what is the baseline.

The reference run is intended to produce output that will be compared with the optimized outputs.
Both reference runs (C++ and MLIR) should also produce similar outputs.
Since the MLIR run needs to be representative of existing Python models, the C++ side has to adapt to have the same ops.

Those runs can be very slow and shouldn't be used for benchmakrs for anything bigger than a single MLP layer.
But they should be used for golden outputs for any models we try to benchmark.
Larger models may need to cache the inputs/outputs to avoid a very slow reference run on every benchmark loop.

### XSMM Runs

These are the ninja optimized code, either by hand in C++ calling libxsmm directly, or by the Tensor Compiler, generating the calls automatically.
The blocking/tiling/fusing parameters are in optimal configuration and should be the fastest runs.

The C++ code represent what the compiler _should_ be doing if it can get the transforms right.
The MLIR code should be within 95% of the performance from C++ runs.

## How to Build

The C++ benchmarks are build by CMake and the binaries will be available in the build directory.
The sources are in `benchmarks/CPPHarness`, each directory under `src` is a separate benchmark.

The MLIR benchmarks use the `MLIRHarness`, which in turn uses `tpp-run` to JIT-compile the MLIR files in `test/Benchmarks` using the default pass pipeline.

## How to Run

There are two ways of running benchmarks: manual and automatic.

### Automatic Runs

There's a Python driver in this folder that, once called, will read the `benchmarks.json` file and run all the benchmarks in there.
This is what the CI does.

This will run both C++ and MLIR versions and will print out the results in order.
The output is semi-formatted, human readable and machine parseable, and you can use that to track timings over time.

You can run it from the `build` directory via `ninja benchmarks`, or you can use the Python script directly for more control.

Use `driver.py -h` for its options.

### Manual Runs

This is for developers to test their transforms in the compiler.

#### C benchmakrs

For C++ benchmarks, run their respective binaries with `-h` to see the options, or look at the `driver.py` script for more options.

The binaries generated on the build directory accept certain arguments to it, for example:
* `-x`: Runs the XSMM version (if available), not the reference one.
* `-n N`: Changes the number of iterations to run.
* `-m MxNxK`: Sets the required shape for the tensors for simpler benchmarks.
* `-r`: Sets the input to be random. If omitted, inputs are constant `all_ones`.
* `-s SEED`: Sets the random seed for the input generator.

Larger benchmarks, for example multiple layers and models, can have multiple tensors (weights, inputs, bias).
For simplicity, weights and biases should be constants in IR (can be random, but as a constant in code), while only inputs can be run-time variable.
In the future, we should support reading input from files and have a more complext configuration (ex. a JSON file, multiple binary files, etc).

#### MLIR benchmakrs

The MLIR benchmarks are also run as tests on a normal test run and are available under `test/Benchmarks`.
The harness (`benchmarks/MLIRHarness/controller.py`) automatically detects tool paths, libraries and even LLVM LIT variables.
It also reads the MLIR file and parsers the FileCheck RUN lines to know how to run both `tpp-opt` and `tpp-run`.

You can use the `-vv` flag, just like the driver, to see what's going on inside, and repeat the steps by hand, if needed.

The flags are the same as the C++ benchmarks.

However, unlike the C benchmarks, it's hard to change the MLIR tensor shapes with a flag, that's why we have multiple MLIR files for a single C++ benchmark.

## How to Add New Runs

To add a new benchmark, you need to add the following items:
 * A new directory in `benchmarks/CPPHarness`.
 * A C++ implementation with a reference (optional) and a libxsmm in that directory.
 * An MLIR file in `test/Benchmarks` with the same kernel, in IR form.
 * Update `benchmarks.json` to add those files.

When in doubt, look at other benchmarks and follow the same steps.
