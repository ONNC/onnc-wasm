#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "benchmark.h"

void ONNC_RUNTIME_batchnormalization_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_X
  ,int32_t input_X_ndim, const int32_t * restrict input_X_dims
  ,const float * restrict input_scale
  ,int32_t input_scale_ndim, const int32_t * restrict input_scale_dims
  ,const float * restrict input_B
  ,int32_t input_B_ndim, const int32_t * restrict input_B_dims
  ,const float * restrict input_mean
  ,int32_t input_mean_ndim, const int32_t * restrict input_mean_dims
  ,const float * restrict input_var
  ,int32_t input_var_ndim, const int32_t * restrict input_var_dims
  ,float * restrict output_Y
  ,int32_t output_Y_ndim, const int32_t * restrict output_Y_dims
  ,float * restrict output_mean
  ,int32_t output_mean_ndim, const int32_t * restrict output_mean_dims
  ,float * restrict output_var
  ,int32_t output_var_ndim, const int32_t * restrict output_var_dims
  ,float * restrict output_saved_mean
  ,int32_t output_saved_mean_ndim, const int32_t * restrict output_saved_mean_dims
  ,float * restrict output_saved_var
  ,int32_t output_saved_var_ndim, const int32_t * restrict output_saved_var_dims
  ,float epsilon
  ,float momentum
  ,int32_t spatial
) {

#ifndef NDEBUG
  host_QITC_time_start("batchnormalization");
#endif // NDEBUG

  // Preparation
  int32_t xN = input_X_dims[0], xC = input_X_dims[1];
  // TODO: spatial
  int32_t strideSize = 1;
  for(int32_t i = 2; i < input_X_ndim; ++i){
    strideSize *= input_X_dims[i];
  }

  for(int32_t iN = 0; iN < xN; ++iN){
    for(int32_t iC = 0; iC < xC; ++iC){
      const float *pIMean = input_mean + iN * xC;
      const float *pIVariance = input_var + iN * xC;
      const float *pX = input_X + iN * xC * strideSize + iC * strideSize;
      float *pY = output_Y + iN * xC * strideSize + iC * strideSize;
      // Output
      for(int32_t i = 0; i < strideSize; ++i){
        pY[i] = input_scale[iC] * (pX[i] - pIMean[iC]) / sqrtf(pIVariance[iC] + epsilon) + input_B[iC];
      }
    }
  }

#ifndef NDEBUG
  host_QITC_time_stop("batchnormalization", "operator batchnormalization");
  host_QITC_time_clear("batchnormalization");
#endif // NDEBUG
}

void ONNC_RUNTIME_batchnormalization_int8(
  void * restrict onnc_runtime_context
  ,const int8_t * restrict input_X
  ,int32_t input_X_ndim, const int32_t * restrict input_X_dims
  ,const int8_t * restrict input_scale
  ,int32_t input_scale_ndim, const int32_t * restrict input_scale_dims
  ,const int8_t * restrict input_B
  ,int32_t input_B_ndim, const int32_t * restrict input_B_dims
  ,const int8_t * restrict input_mean
  ,int32_t input_mean_ndim, const int32_t * restrict input_mean_dims
  ,const int8_t * restrict input_var
  ,int32_t input_var_ndim, const int32_t * restrict input_var_dims
  ,int8_t * restrict output_Y
  ,int32_t output_Y_ndim, const int32_t * restrict output_Y_dims
  ,int8_t * restrict output_mean
  ,int32_t output_mean_ndim, const int32_t * restrict output_mean_dims
  ,int8_t * restrict output_var
  ,int32_t output_var_ndim, const int32_t * restrict output_var_dims
  ,int8_t * restrict output_saved_mean
  ,int32_t output_saved_mean_ndim, const int32_t * restrict output_saved_mean_dims
  ,int8_t * restrict output_saved_var
  ,int32_t output_saved_var_ndim, const int32_t * restrict output_saved_var_dims
  ,int8_t epsilon
  ,int8_t momentum
  ,int32_t spatial
) {

#ifndef NDEBUG
  host_QITC_time_start("batchnormalization");
#endif // NDEBUG

  // Preparation
  int32_t xN = input_X_dims[0], xC = input_X_dims[1];
  // TODO: spatial
  int32_t strideSize = 1;
  for(int32_t i = 2; i < input_X_ndim; ++i){
    strideSize *= input_X_dims[i];
  }

  for(int32_t iN = 0; iN < xN; ++iN){
    for(int32_t iC = 0; iC < xC; ++iC){
      const int8_t *pIMean = input_mean + iN * xC;
      const int8_t *pIVariance = input_var + iN * xC;
      const int8_t *pX = input_X + iN * xC * strideSize + iC * strideSize;
      int8_t *pY = output_Y + iN * xC * strideSize + iC * strideSize;
      // Output
      for(int32_t i = 0; i < strideSize; ++i){
        pY[i] = input_scale[iC] * (pX[i] - pIMean[iC]) / sqrtf(pIVariance[iC] + epsilon) + input_B[iC];
      }
    }
  }

#ifndef NDEBUG
  host_QITC_time_stop("batchnormalization", "operator batchnormalization");
  host_QITC_time_clear("batchnormalization");
#endif // NDEBUG
}