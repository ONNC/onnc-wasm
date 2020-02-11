#include <stdint.h>
typedef int32_t ONNC_INDEX_TYPE;

#include "generic/size.h"
#include <string.h>

#include "benchmark.h"

void ONNC_RUNTIME_concat_float(void* restrict context,
    const float* restrict* inputs, int32_t count, const int32_t* restrict ndims, const int32_t* restrict* shapes,
    float* restrict output, int32_t ndim, const int32_t* restrict shape, int32_t axis)
{

#ifndef NDEBUG
  host_QITC_time_start("concat");
#endif // NDEBUG

    if (count) {
        int32_t slices = onnc_size(shapes[0], axis);
        int32_t block = onnc_size(shapes[0] + axis + 1, ndims[0] - axis - 1);

        for (int32_t j = 0; j < slices; ++j) {
            for (int32_t i = 0; i < count; ++i) {
                int32_t length = shapes[i][axis] * block;
                memcpy(output, inputs[i] + j * length, length * sizeof(float));
                output += length;
            }
        }
    }

#ifndef NDEBUG
  host_QITC_time_stop("concat", "operator concat");
  host_QITC_time_clear("concat");
#endif // NDEBUG
}
