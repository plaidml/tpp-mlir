#pragma once

#include "llvm/Support/CommandLine.h"
#include <chrono>
#include <numeric>
#include <string>
#include <vector>

namespace {
/// Command line options in a struct, to prevent the need for global static
/// initializer, used by the ArgParser, when invoked.
struct BenchOptions {
  llvm::cl::opt<std::string> inputDims{
      "input", llvm::cl::desc("Input dimensions, 1/2/3/4D"),
      llvm::cl::value_desc("MxNxKxB"), llvm::cl::init("")};
  llvm::cl::opt<bool> xsmm{"xsmm", llvm::cl::desc("Run XSMM optimized version"),
                           llvm::cl::init(false)};
  llvm::cl::opt<int> iter{
      "iter",
      llvm::cl::desc("Number of iterations (default based on input size)"),
      llvm::cl::value_desc("N"), llvm::cl::init(0)};
  llvm::cl::opt<bool> random{"random",
                             llvm::cl::desc("Randomize input (0.0 ~ 1.0)"),
                             llvm::cl::init(false)};
  llvm::cl::opt<int> seed{"seed",
                          llvm::cl::desc("Random seed (default localtime)"),
                          llvm::cl::value_desc("N"), llvm::cl::init(0)};
  llvm::cl::opt<bool> verbose{"v", llvm::cl::desc("Verbose"),
                              llvm::cl::init(false)};
  llvm::cl::opt<bool> gflops{
      "gflops", llvm::cl::desc("Emit results in GFLOP/s (default is ms)"),
      llvm::cl::init(false)};
};
} // namespace

/// Argument parser and configuration for benchmarks. Sets the variables to the
/// defaults, overridden from the command line and potential internal logic
/// later on.
///
/// Defaults overriden:
///  * input: convert string format to array of integers.
///  * iter: if 0, calculate in terms of MxNxKxB.
///  * seed: if 0, take localtime as integer.
///
/// Number of iterations will be calculated per matmul, so if a kernel has
/// many matmuls (multiple layers), then it should divide that number by the
/// number of layers.
struct BenchConfig {
  /// Define what an "average" matrix size is (fiddle for best default iter)
  const size_t averageMatrixSize = 128 * 256 * 512;
  /// Default number of iterations if nothing else is available
  /// This is for a single layer, larger kernels need to divide this
  const size_t defaultIter = 100;

  /// Input dimensions
  std::vector<size_t> dims;
  /// True if run is XSMM
  bool xsmm;
  /// Number of iterations
  int iter;
  /// True if input is random
  bool random;
  /// Random seed
  size_t seed;
  /// Verbose run info
  bool verbose;
  /// GFLOPs results
  bool gflops;

  BenchConfig(int argc, char **argv) {
    BenchOptions options;
    llvm::cl::ParseCommandLineOptions(argc, argv, "TPP-MLIR C++ Benchmark\n");

    // Simple settings
    xsmm = options.xsmm;
    random = options.random;
    verbose = options.verbose;
    gflops = options.gflops;

    // Parse input dims
    size_t m[4];
    int found = sscanf(options.inputDims.c_str(), "%ldx%ldx%ldx%ld", &m[0],
                       &m[1], &m[2], &m[3]);
    if (found > 0)
      dims.insert(dims.begin(), m, m + found);

    // Override iter, if zero
    iter = options.iter;
    if (!iter) {
      // For an "average size" matrix multiply, this should be enough
      iter = defaultIter;
      if (dims.size() != 0) {
        // Divide the "average size" by the actual size to get the ratio
        float size = std::accumulate(dims.begin(), dims.end(), 1.0,
                                     std::multiplies<float>());
        float ratio = averageMatrixSize / size;
        iter *= ratio;
      }
    }

    // Override seed, if zero
    seed = options.seed;
    if (random && !seed) {
      seed = clock();
    }
  }
};
