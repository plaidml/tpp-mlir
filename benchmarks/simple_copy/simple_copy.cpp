#include "Config.h"
#include "Bench.h"
#include "Tensor.h"

#include <iomanip>
#include <iostream>

template <class T>
struct SimpleCopyKernel : public KernelInterface<T> {
  void runRef(std::vector<Tensor<T>>& args) override {
    assert(args.size() == 2 && "wrong rank for copy");
    // Uses copy constructor to copy data
    auto& input = args[0];
    auto& output = args[1];
    output = input;
  }

  void runXSMM(std::vector<Tensor<T>>& args) override {
    // TODO: Call libxsmm copy
    assert(0 && "XSMM kernel not implemented yet");
  }
};

int main(int argc, char **argv) {
  // Assume success until proven wrong
  int returnValue = 0;

  // These need to be the same as the MLIR file
  unsigned m = 1024;
  unsigned n = 1024;

  // Cmd-line args
  BenchConfig config(argc, argv);
  if (config.dims.size() == 2) {
    m = config.dims[0];
    n = config.dims[1];
  }
  if (config.verbose) {
    std::cerr << "Kernel version: " << ((config.xsmm) ? "xsmm" : "ref") << std::endl;
    std::cerr << "[ " << m << ", " << n << " ] X " << config.iter << std::endl;
  }

  // Init benchmark (TODO: support BF16)
  double gflops = (double)(n*m)/1e9;
  auto bench = Benchmark<SimpleCopyKernel<float>, float>(config.iter, gflops, config.xsmm);
  bench.setArg({ConstantTensor<float>({m, n}), EmptyTensor<float>({m, n})});

  auto& input = bench.getArg(0);
  auto& output = bench.getArg(1);

  // Warmup
  bench.warmup();

  if (input != output) {
    std::cerr << "Comparison failed" << std::endl;
    returnValue = 1;
  }

  // Run the reference benchmark
  bench.run();

  double mean = bench.getMean();
  double stdev = bench.getStdev();
  std::string unit = "ms";
  if (gflops)
    unit = "gflops";

  std::cout << std::fixed << std::setw(9) << std::setprecision(3) <<
    mean << " +- " << stdev << " " << unit << std::endl;

  return returnValue;
}
