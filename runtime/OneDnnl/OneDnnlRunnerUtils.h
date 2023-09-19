#ifndef ONE_DNNL_EXECUTIONENGINE_CRUNNERUTILS_H
#define ONE_DNNL_EXECUTIONENGINE_CRUNNERUTILS_H

#include "mlir/ExecutionEngine/RunnerUtils.h"

extern "C" MLIR_RUNNERUTILS_EXPORT void
linalg_matmul_blas(size_t m, size_t n, size_t k, const float *A, size_t offsetA,
                   size_t lda, const float *B, size_t offsetB, size_t ldb,
                   float *C, size_t offsetC, size_t ldc);

#endif // ONE_DNNL_EXECUTIONENGINE_CRUNNERUTILS_H
