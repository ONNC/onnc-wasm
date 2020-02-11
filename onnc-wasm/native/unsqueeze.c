#include <stdint.h>
#include <stdbool.h>

#include "benchmark.h"

void ONNC_RUNTIME_unsqueeze_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_data
  ,int32_t input_data_ndim, const int32_t * restrict input_data_dims
  ,float * restrict output_expanded
  ,int32_t output_expanded_ndim, const int32_t * restrict output_expanded_dims
  ,int32_t * restrict axes
  ,int32_t number_of_axes
) {

#ifndef NDEBUG
  host_QITC_time_start("unsqueeze");
#endif // NDEBUG

  int32_t size_N = 1;
  for(int32_t i = 0; i < input_data_ndim; ++i) {
    size_N *= input_data_dims[i];
  }
  for(int32_t i = 0; i < size_N; ++i) {
    output_expanded[i] = input_data[i];
  }

#ifndef NDEBUG
  host_QITC_time_stop("unsqueeze", "operator unsqueeze");
  host_QITC_time_clear("unsqueeze");
#endif // NDEBUG
}
