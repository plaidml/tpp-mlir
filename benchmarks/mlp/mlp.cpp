#include "Config.h"
#include "Bench.h"
#include "Tensor.h"

#include <iomanip>
#include <iostream>

/* Reference implementation of a mlp */
// A = 128 x 256
// B = 256 x 512
// C =   1 x 512
// O = 128 x 512
// Total flops = broadcast O(n*m) + matmul O(n*m*k) + ReLU (O(n*m)
// 128x512x2 (131072) + 128x256x512 (16777216) = 16908288

template <class T>
struct MLPKernel : public KernelInterface<T> {
  void runRef(std::vector<Tensor<T>>& args) override {
    assert(args.size() == 4 && "wrong rank for MLP");
    auto& a = args[0];
    auto& b = args[1];
    auto& c = args[2];
    auto& o = args[3];

    int m = o.getDim(0);
    int n = o.getDim(1);
    int k = a.getDim(1);

    // BROADCAST INIT O from C
    for (int mi = 0; mi < m; ++mi) {
      for (int ni = 0; ni < n; ++ni) {
        o[mi*n+ni] = c[ni];
      }
    }

    // MATMUL O += A x B
    for (int mi = 0; mi < m; ++mi) {
      for (int ni = 0; ni < n; ++ni) {
        for (int ki = 0; ki < k; ++ki) {
          o[mi*n+ni] += a[mi*k+ki] * b[ki*n+ni];
        }
      }
    }

    // RELU
    for (int ni = 0; ni < n; ++ni) {
      for (int mi = 0; mi < m; mi++) {
        o[mi*n+ni] = std::max(0.0f, o[mi*n+ni]);
      }
    }
  }

  void runXSMM(std::vector<Tensor<T>>& args) override {
    // TODO: Call libxsmm copy
    assert(0 && "XSMM kernel not implemented yet");
  }
};

int main(int argc, char *argv[]) {
  // Assume success until proven wrong
  int returnValue = 0;

  // These need to be the same as the MLIR file
  unsigned m = 128;
  unsigned n = 512;
  unsigned k = 256;

  // Cmd-line args
  BenchConfig config(argc, argv);
  if (config.dims.size() == 3) {
    m = config.dims[0];
    n = config.dims[1];
    k = config.dims[2];
  }
  if (config.verbose) {
    std::cerr << "Kernel version: " << ((config.xsmm) ? "xsmm" : "ref") << std::endl;
    std::cerr << "[ " << m << ", " << n << " ] = "
              << "[ " << m << ", " << k << " ] * "
              << "[ " << k << ", " << n << " ] X " << config.iter << std::endl;
  }

  // Init benchmark (TODO: support BF16)
  double gflops = static_cast<double>((n*m) + (2*n*m*k) + (n*m)) / 1e9;
  auto bench = Benchmark<MLPKernel<float>, float>(config.iter, gflops, config.xsmm);
  bench.setArg({ConstantTensor<float>({m, k}),
                ConstantTensor<float>({k, n}),
                ConstantTensor<float>({n}),
                EmptyTensor<float>({m, n})});

  // Warmup (TODO: Check output)
  bench.warmup();

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
