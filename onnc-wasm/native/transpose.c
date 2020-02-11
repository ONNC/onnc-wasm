#include <stdint.h>
typedef int32_t ONNC_INDEX_TYPE;

#include "generic/transpose.h"

#include "benchmark.h"

void ONNC_RUNTIME_transpose_float(void* restrict context,
    const float* restrict X, int32_t Xdim, const int32_t* restrict Xshape,
    float* restrict Y, int32_t Ydim, const int32_t* restrict Yshape,
    const int32_t* restrict permute, int32_t dim)
{
#ifndef NDEBUG
  host_QITC_time_start("transpose");
#endif // NDEBUG

  ONNC_TRANSPOSE(float, Y, X, permute, Xshape, Xdim);

#ifndef NDEBUG
  host_QITC_time_stop("transpose", "operator transpose");
  host_QITC_time_clear("transpose");
#endif // NDEBUG
}
