#include <stdint.h>
#include <stdbool.h>

#include "benchmark.h"

void ONNC_RUNTIME_relu_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_X
  ,int32_t input_X_ndim, const int32_t * restrict input_X_dims
  ,float * restrict output_Y
  ,int32_t output_Y_ndim, const int32_t * restrict output_Y_dims
  
) {

#ifndef NDEBUG
  host_QITC_time_start("relu");
#endif // NDEBUG

  int32_t size = 1;

	for(int32_t i = 0 ; i < input_X_ndim ; ++i){
		size *= input_X_dims[i];
	}

	for(int32_t i = 0 ; i < size ; ++i){
	    float tmp_val = input_X[i];
		output_Y[i] = (tmp_val >= 0.0f) ? tmp_val : 0.0f;
	}

#ifndef NDEBUG
  host_QITC_time_stop("relu", "operator relu");
  host_QITC_time_clear("relu");
#endif // NDEBUG
}

void ONNC_RUNTIME_relu_int8(
  void * restrict onnc_runtime_context
  ,const int8_t * restrict input_X
  ,int32_t input_X_ndim, const int32_t * restrict input_X_dims
  ,int8_t * restrict output_Y
  ,int32_t output_Y_ndim, const int32_t * restrict output_Y_dims
) {

#ifndef NDEBUG
  host_QITC_time_start("relu");
#endif // NDEBUG

  int32_t size = 1;

	for(int32_t i = 0 ; i < input_X_ndim ; ++i){
		size *= input_X_dims[i];
	}

	for(int32_t i = 0 ; i < size ; ++i){
    int8_t tmp_val = input_X[i];
		output_Y[i] = (tmp_val >= 0.0f) ? tmp_val : 0.0f;
	}

#ifndef NDEBUG
  host_QITC_time_stop("relu", "operator relu");
  host_QITC_time_clear("relu");
#endif // NDEBUG
}