#include <stdint.h>
typedef int32_t ONNC_INDEX_TYPE;

#include "generic/assign.h"
#include "generic/binary.h"
#include "generic/size.h"
#include <string.h>

#include "benchmark.h"

static float sum_(float a, float b)
{
    return a + b;
}

void ONNC_RUNTIME_sum_float(void* restrict context,
    const float* restrict* inputs, int32_t count, const int32_t* restrict ndims, const int32_t* restrict* shapes,
    float* restrict output, int32_t ndim, const int32_t* restrict shape)
{
#ifndef NDEBUG
    host_QITC_time_start("sum");
#endif // NDEBUG

    if (count) {
        ONNC_ASSIGN(float, output, shape, ndim, inputs[0], shapes[0], ndims[0]);

        for (int32_t i = 1; i < count; ++i)
            ONNC_BINARY(float, output, shape, ndim, inputs[i], shapes[i], ndims[i], sum_);
    }
    else {
        memset(output, 0, onnc_size(shape, ndim) * sizeof(float));
    }

#ifndef NDEBUG
    host_QITC_time_stop("sum", "operator sum");
    host_QITC_time_clear("sum");
#endif // NDEBUG
}
