#ifndef ONE_DNNL_EXECUTIONENGINE_CRUNNERUTILS_H
#define ONE_DNNL_EXECUTIONENGINE_CRUNNERUTILS_H

#include "mlir/ExecutionEngine/RunnerUtils.h"

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_linalg_matmul_view64x64xf32_view64x64xf32_view64x64xf32(
    StridedMemRefType<float, 2> *, StridedMemRefType<float, 2> *,
    StridedMemRefType<float, 2> *);

#endif // ONE_DNNL_EXECUTIONENGINE_CRUNNERUTILS_H
