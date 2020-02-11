#include <stdint.h>
#include <stdbool.h>

#include "benchmark.h"

void ONNC_RUNTIME_reshape_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_data
  ,int32_t input_data_ndim, const int32_t * restrict input_data_dims
  ,const float * restrict input_shape
  ,int32_t input_shape_ndim, const int32_t * restrict input_shape_dims
  ,float * restrict output_reshaped
  ,int32_t output_reshaped_ndim, const int32_t * restrict output_reshaped_dims
  
) {

#ifndef NDEBUG
    host_QITC_time_start("reshape");
#endif // NDEBUG

    int32_t size = 1;
    for(int32_t dim = 0 ; dim < input_data_ndim ; dim++){
        size *= input_data_dims[dim];
    }

    for(int32_t index = 0 ; index < size ; index++){
        output_reshaped[index] = input_data[index];
    }

#ifndef NDEBUG
  host_QITC_time_stop("reshape", "operator reshape");
  host_QITC_time_clear("reshape");
#endif // NDEBUG
}
