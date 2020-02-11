#include <stdint.h>
typedef int32_t ONNC_INDEX_TYPE;

#include "generic/assign.h"
#include "generic/gemm.h"
#include "benchmark.h"

void ONNC_RUNTIME_gemm_float(void* restrict context,
    const float* restrict A, int32_t Adim, const int32_t* restrict Ashape,
    const float* restrict B, int32_t Bdim, const int32_t* restrict Bshape,
    const float* restrict C, int32_t Cdim, const int32_t* restrict Cshape,
    float* restrict Y, int32_t Ydim, const int32_t* restrict Yshape,
    float alpha, float beta, int32_t transA, int32_t transB)
{
    int32_t rows = Yshape[0];
    int32_t cols = Yshape[1];
    int32_t depth = Ashape[!transA];

#ifndef NDEBUG
    host_QITC_time_start("gemm");
#endif // NDEBUG

    ONNC_ASSIGN(float, Y, Yshape, 2, C, Cshape, Cdim);

    for (int32_t i = 0; i < rows * cols; ++i)
        Y[i] *= beta;

    ONNC_GEMM(float, Y, A, B, alpha, rows, cols, depth, transA, transB);

#ifndef NDEBUG
    host_QITC_time_stop("gemm", "operator gemm");
    host_QITC_time_clear("gemm");
#endif // NDEBUG
}
