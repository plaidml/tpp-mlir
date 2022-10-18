# TPP Runner

This is basically a copy of `mlir-cpu-runner`, using `JitRunnerMain` and its call backs.

The main difference is that we add a wrapper function to call the kernel (entry) function to allow for benchmarking.

However, we'll need to add:
 * Random initialization of input tensors (and control the seed, for stable reproduction)
 * Timing strategy (perhaps with a timer dialect?) to only check kernel times
 * Repeat functionality, to run just the kernel many times, for statistical significance
 * Warm-up calls, to allow for JIT compilation to finish before we start timing

We may have to create new callbacks and change LLVM upstream to do that, though.

## Functionality

The main function of this modified TPP runner is to:
 * (TODO) Automatically find the TPP, LIBXSMM and OpenMP libraries (now is via cmd-line)
 * Find the kernel function and:
   * Discover its arguments (input and output)
   * Allocate memrefs for all and random-initialise the inputs
 * Compile the kernel MLIR module with `main`, `return` and kernel functions
 * Run the main function, gathering the outputs (tensors)
 * Run the kernel function, passing those tensors as arguments
   * Run multiple times and output statistics in benchmark mode
 * Run the `return` function to print/cleanup the results

## Implementation

First approach is to use the existing callbacks to wrap our needed functionality:
 * `mlirTransformer` for parsing the kernel's arguments, preparing the input and calling the kernel, printing the results
 * `llvmModuleBuilder` for finding the libraries and making sure the IR is correct

We'll probably need a way to measure execution of the kernel, not the tensor preparation and print. 
This could be done in IR (via `mlirTransformer`) or via some other callback.

If we add new callbacks, we must upstream this.
