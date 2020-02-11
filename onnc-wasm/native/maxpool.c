#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include "benchmark.h"

static inline bool next_dim(int32_t ndim, int32_t * restrict dim,
                            const int32_t * restrict dim_max) {
  do {
    ndim = ndim - 1;
    dim[ndim] += 1;
    if (dim[ndim] < dim_max[ndim]) {
      return true;
    } else { // reach dimension max
      if (ndim == 0) { // all dimension done
        return false;
      }
      dim[ndim] = 0;
    }
  } while(true);
}

static inline int64_t dim_to_offset(int32_t ndim, const int32_t * restrict dim,
                                    const int32_t * restrict dim_max) {
  int64_t offset = 0;
  int64_t step = 1;
  for (int32_t i = ndim - 1; i >= 0; --i) {
    offset += dim[i] * step;
    step *= dim_max[i];
  }
  return offset;
}

// If it is outside the bounds of the input, use 0.
#define get_value_or_zero(T) \
static inline T get_value_or_zero_ ## T (int32_t ndim, const int32_t * restrict dim_max, \
                                      const T * restrict value, const int32_t * restrict dim) { \
  for (int32_t i = 0; i < ndim; ++i) { \
    if (dim[i] < 0 || dim[i] >= dim_max[i]) { \
      return 0; \
    } \
  } \
  return value[dim_to_offset(ndim, dim, dim_max)]; \
}
get_value_or_zero(float)
get_value_or_zero(int8_t)

void ONNC_RUNTIME_maxpool_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_X
  ,int32_t input_X_ndim, const int32_t * restrict input_X_dims
  ,float * restrict output_Y
  ,int32_t output_Y_ndim, const int32_t * restrict output_Y_dims
  ,float * restrict output_Indices
  ,int32_t output_Indices_ndim, const int32_t * restrict output_Indices_dims
  ,const char * restrict auto_pad
  ,int32_t * restrict kernel_shape
  ,int32_t number_of_kernel_shape
  ,int32_t * restrict pads
  ,int32_t number_of_pads
  ,int32_t storage_order
  ,int32_t * restrict strides
  ,int32_t number_of_strides
) {

#ifndef NDEBUG
  host_QITC_time_start("maxpool");
#endif // NDEBUG

	assert(input_X_ndim == output_Y_ndim);
	int32_t ndim = input_X_ndim;

  int32_t o_dim[ndim];
  memset(o_dim, 0, sizeof(o_dim));
  do { // while o_dim
    int32_t base_dim[ndim];
    for (int32_t i = 2; i < ndim; ++i) {
      base_dim[i] = o_dim[i] * strides[i - 2] - pads[i - 2];
    }

    float max = -FLT_MAX;

    int32_t k_dim[ndim - 2];
    memset(k_dim, 0, sizeof(k_dim));
    do { // while k_dim
      int32_t i_dim[ndim];
      i_dim[0] = o_dim[0]; // N
      i_dim[1] = o_dim[1]; // C
      for (int32_t i = 2; i < ndim; ++i) {
        i_dim[i] = base_dim[i] + k_dim[i - 2];
      }
      float input = get_value_or_zero_float(ndim, input_X_dims, input_X, i_dim);
      max = fmaxf(input, max);
    } while (next_dim(ndim - 2, k_dim, kernel_shape));

    output_Y[dim_to_offset(ndim, o_dim, output_Y_dims)] = max;
  } while (next_dim(ndim, o_dim, output_Y_dims));

#ifndef NDEBUG
  host_QITC_time_stop("maxpool", "operator maxpool");
  host_QITC_time_clear("maxpool");
#endif // NDEBUG
}

void ONNC_RUNTIME_maxpool_int8(
  void * restrict onnc_runtime_context
  ,const int8_t * restrict input_X
  ,int32_t input_X_ndim, const int32_t * restrict input_X_dims
  ,int8_t * restrict output_Y
  ,int32_t output_Y_ndim, const int32_t * restrict output_Y_dims
  ,int8_t * restrict output_Indices
  ,int32_t output_Indices_ndim, const int32_t * restrict output_Indices_dims
  ,const char * restrict auto_pad
  ,int32_t * restrict kernel_shape
  ,int32_t number_of_kernel_shape
  ,int32_t * restrict pads
  ,int32_t number_of_pads
  ,int32_t storage_order
  ,int32_t * restrict strides
  ,int32_t number_of_strides
) {

#ifndef NDEBUG
  host_QITC_time_start("maxpool");
#endif // NDEBUG

	assert(input_X_ndim == output_Y_ndim);
	int32_t ndim = input_X_ndim;

  int32_t o_dim[ndim];
  memset(o_dim, 0, sizeof(o_dim));
  do { // while o_dim
    int32_t base_dim[ndim];
    for (int32_t i = 2; i < ndim; ++i) {
      base_dim[i] = o_dim[i] * strides[i - 2] - pads[i - 2];
    }

    int8_t max = INT8_MIN;

    int32_t k_dim[ndim - 2];
    memset(k_dim, 0, sizeof(k_dim));
    do { // while k_dim
      int32_t i_dim[ndim];
      i_dim[0] = o_dim[0]; // N
      i_dim[1] = o_dim[1]; // C
      for (int32_t i = 2; i < ndim; ++i) {
        i_dim[i] = base_dim[i] + k_dim[i - 2];
      }
      int8_t input = get_value_or_zero_int8_t(ndim, input_X_dims, input_X, i_dim);
      max = fmaxf(input, max);
    } while (next_dim(ndim - 2, k_dim, kernel_shape));

    output_Y[dim_to_offset(ndim, o_dim, output_Y_dims)] = max;
  } while (next_dim(ndim, o_dim, output_Y_dims));

#ifndef NDEBUG
  host_QITC_time_stop("maxpool", "operator maxpool");
  host_QITC_time_clear("maxpool");
#endif // NDEBUG
}
