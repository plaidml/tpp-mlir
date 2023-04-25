# TPP MLIR Benchmarks

This directory contains benchmarks comparing TPP-MLIR with LIBXSMM-DNN.
This allows us to measure how much off are we when compared with ninja-written code.

## Run Types

There are two types of runs: TPP-MLIR (suffix `_mlir`) and XSMM-DNN (suffic `_dnn`).
Each type can choose a number of options, environment variables and CPU flag support.

Common options are:
 * Use of OpenMP (via `OMP_NUM_THREADS` in environment)
 * Increase iterations (via `-n` in MLIR runs or first argument in DNN runs)
 * Optimization flags (via `-run-args=` in `flags` for MLIR runs)
 * Restrict support to BF16 (via `avx512.*` or `svebf16` in `extensions`)

## How to Run

There are two ways of running benchmarks: manual and automatic.

### Automatic Runs

Running `ninja benchmarks` in the build directory will run the CI JSON file `benchmarks.json` in the `benchmarks` folder.

There's a Python driver in this folder that, once called, will read the `benchmarks.json` file and run all the benchmarks in there.

This is what the CMake target `benchmarks` does.

This will run all benchmarks inside that JSON file and will print out the results in order.
The output is semi-formatted, human readable and machine parseable, and you can use that to track timings over time.

If you have more than one build directory (release/debug/etc), be sure to specify which you want to use with the `--build` flag.
Use `driver.py -h` for its options.

There are more JSON files with more extended benchmarks in the `benchmarks` directory, which can be used with the `driver.py` script in the same way.

### Manual Runs

This is for developers to test their transforms in the compiler.

#### MLIR Benchmarks

The MLIR benchmarks are also run as tests on a normal test run and are available under `benchmarks/mlir`.
The harness (`benchmarks/harness/controller.py`) automatically detects tool paths, libraries and even LLVM LIT variables.
It also reads the MLIR file and parsers the FileCheck RUN lines to know how to run both `tpp-opt` and `tpp-run`.

To see the harness' command line, run the driver with `-vv` and it will dump all command lines for both MLIR and XSMM-DNN runs.
For MLIR runs, through the harness, you can also use the `-vv` flag to see what's going on inside, and repeat the steps by hand, if needed.

Unlike the XSMM-DNN benchmarks, it's hard to change the MLIR tensor shapes with a flag, that's why we have multiple MLIR files for a single C++ benchmark.

## How to Add New Runs

To add a new benchmark, you need to add the following items:
 * Add the XSMM-DNN run that simulates the kernel you're tryig to run.
   * If not building yet, add CMake recipes to build it from libxsmm-dnn sources.
   * If not available in libxsmm-dnn yet, make sure to add a new program there first.
 * An MLIR file in `benchmarks/mlir` with the same kernel, in IR form.
 * Update `benchmarks.json` (or some other) to add those files.

When in doubt, look at other benchmarks and follow the same steps.
