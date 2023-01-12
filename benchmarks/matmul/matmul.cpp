#include "Config.h"
#include "Bench.h"
#include "Tensor.h"

//#include <libxsmm.h>

#include <iomanip>
#include <iostream>

template <class T>
struct MatmulKernel : public KernelInterface<T> {
  void runRef(std::vector<Tensor<T>>& args) override {
    assert(args.size() == 3 && "wrong rank for MLP");
    auto& a = args[0];
    auto& b = args[1];
    auto& o = args[2];

    int m = o.getDim(0);
    int n = o.getDim(1);
    int k = a.getDim(1);

    // MATMUL O += A x B
    for (int mi = 0; mi < m; ++mi) {
      for (int ni = 0; ni < n; ++ni) {
        for (int ki = 0; ki < k; ++ki) {
          o[mi*n+ni] += a[mi*k+ki] * b[ki*n+ni];
        }
      }
    }
  }

  void runXSMM(std::vector<Tensor<T>>& args) override {
    // TODO: Call libxsmm copy
    assert(0 && "XSMM kernel not implemented yet");
    /*
    void matmul_xsmm(DECL_VEC2D_FUNC_IN_ARGS(a, float),
                DECL_VEC2D_FUNC_IN_ARGS(b, float),
                DECL_VEC2D_FUNC_OUT_ARGS(c, float)) {
      const libxsmm_blasint lda = (libxsmm_blasint)b_stride0;
      const libxsmm_blasint ldb = (libxsmm_blasint)a_stride0;
      const libxsmm_blasint ldc = (libxsmm_blasint)c_stride0;
      const libxsmm_blasint m = (libxsmm_blasint)c_size1;
      const libxsmm_blasint n = (libxsmm_blasint)c_size0;
      const libxsmm_blasint k = (libxsmm_blasint)a_size1;
    #if defined(NO_JIT)
      const float alpha = 1.f, beta = 1.f;
      LIBXSMM_INLINE_XGEMM(float, float, "N", "N", &m, &n, &k, &alpha, b_alignedPtr,
                           &lda, a_alignedPtr, &ldb, &beta, c_alignedPtr, &ldc);
    #else
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
          m, n, k, lda, ldb, ldc, LIBXSMM_DATATYPE(float), LIBXSMM_DATATYPE(float),
          LIBXSMM_DATATYPE(float), LIBXSMM_DATATYPE(float));
      libxsmm_xmmfunction kernel = {NULL};
      kernel.gemm = libxsmm_dispatch_gemm_v2(gemm_shape, LIBXSMM_GEMM_FLAG_NONE,
                                             LIBXSMM_PREFETCH_NONE);
      libxsmm_gemm_param gemm_param;
      // memset(&gemm_param, 0, sizeof(gemm_param));
      gemm_param.a.primary = (void *)b_alignedPtr;
      gemm_param.b.primary = (void *)a_alignedPtr;
      gemm_param.c.primary = c_alignedPtr;
      // no prefetch (gemm_param.?.quaternary not used)
      kernel.gemm(&gemm_param);
    #endif
    } */
  }
};

int main(int argc, char *argv[]) {
  // Assume success until proven wrong
  int returnValue = 0;

  // These need to be from the command line
  unsigned m = 0;
  unsigned n = 0;
  unsigned k = 0;

  // Cmd-line args
  BenchConfig config(argc, argv);
  if (config.dims.size() == 3) {
    m = config.dims[0];
    n = config.dims[1];
    k = config.dims[2];
  } else {
    std::cerr << "--input argument required to be 3D, use --help for options\n";
    return 1;
  }

  if (config.verbose) {
    std::cerr << "Kernel version: " << ((config.xsmm) ? "xsmm" : "ref") << std::endl;
    std::cerr << "[ " << m << ", " << n << " ] = "
              << "[ " << m << ", " << k << " ] * "
              << "[ " << k << ", " << n << " ] X " << config.iter << std::endl;
  }

  // Init benchmark (TODO: support BF16)
  double gflops = static_cast<double>(2*n*m*k) / 1e9;
  auto bench = Benchmark<MatmulKernel<float>, float>(config.iter, gflops, config.xsmm);
  bench.setArg({ConstantTensor<float>({m, k}),
                ConstantTensor<float>({k, n}),
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
