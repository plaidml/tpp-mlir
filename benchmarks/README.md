# TPP MLIR Benchmarks

This directory contains benchmarks comparing MLIR passes of our compiler with manual implementaitons written in C++ calling libxsmm directly.
This allows us to measure how much off are we when compared with ninja-written code.

## Run Types

There are three types of runs: reference, tpp-mlir and libxsmm.

### Reference Runs

These runs pure C without using TPP/XSMM, just executing the code as is, with compiler at -O3.
This is the most naive and inefficient way of running the kernels and lets us know what is the baseline.

### TPP-MLIR Runs

These take in the MLIR implementation and run TPP optimizations on them before running the kernels.
The timings of these runs must be between the vanilla runs and the ninja runs, closer to the latter.
We aim to be within 90% of the performance from ninja runs.

### Ninja Runs

These do same computations as the MLIR kernels, but calling libxsmm directly.
The blocking/tiling/fusing parameters are in optimal configuration and should be the fastest runs.
They also represent what the compiler _should_ be doing if it can get the transforms right.

## How to Run

There are two ways of running benchmarks: manual and automatic.

### Automatic Runs

There's a Python driver in this folder that, once called, will read the `benchmarks.json` file and run all the benchmarks in there.
This is what the CI does.

This will run both C++ and MLIR versions and will print out the results in order.
The output is semi-formatted, human readable and machine parseable, and you can use that to track timings over time.

Use `driver.py -h` for its options.

### Manual Runs

This is for developers to test their transforms in the compiler.

The driver above does two things:
1. Compiles the C++ file and run it with known options, which print results.
2. Call the benchmark harness on the MLIR file, which runs it and print results.

You can call the driver with the `-v` (or `-vv`) option to see logs, and use those to repeat the runs by hand.
Since the driver (and the harness) detect include/library/tools paths, it's wise to use it before trying it by hand.

#### C benchmakrs

The C benchmakrs are split into two stages: compile and run.

The compilation is assuming `clang` is in the path and picking the other flags from the repository.
You can get those options from a `./driver.py -vv` run.

Once compiled, the binary is generated on the same directory as the source and you can pass certain arguments to it, for example:
* `-x`: Runs the XSMM version (if available), not the reference one.
* `-n`: Changes the number of iterations to run

Some benchmarks, for example the matmul, have also an argument to define the size of the GEMM in the format `NxMxK`.

#### MLIR benchmakrs

The MLIR benchmarks are also run as tests on a normal test run and are available under `test/Benchmarks`.
The harness (`benchmarks/harness/controller.py`) automatically detects tool paths, libraries and even LLVM LIT variables.
It also reads the MLIR file and parsers the FileCheck RUN lines to know how to run both `tpp-opt` and `tpp-run`.

You can use the `-vv` flag, just like the driver, to see what's going on inside, and repeat the steps by hand, if needed.

Unlike the C benchmarks, it's hard to change the MLIR tensor shapes with a flag, that's why we have multiple MLIR files for a single C benchmark.

## How to Add New Runs

To add a new benchmark, you need to add the following items:
 * A new directory in `benchmarks`.
 * A C implementation with a reference (optional) and a libxsmm in that directory.
 * An MLIR file in `test/Benchmarks` with the same kernel, in IR form.
 * Update `benchmarks.json` to add those files.

When in doubt, look at other benchmarks and follow the same steps.
