#pragma once

#include <stdint.h>

void ONNC_RUNTIME_add_float(
  void * onnc_runtime_context
  ,const float * input_A
  ,int32_t input_A_ndim, const int32_t * input_A_dims
  ,const float * input_B
  ,int32_t input_B_ndim, const int32_t * input_B_dims
  ,float * output_C
  ,int32_t output_C_ndim, const int32_t * output_C_dims
);

void ONNC_RUNTIME_averagepool_float(
  void * onnc_runtime_context
  ,const float * input_X
  ,int32_t input_X_ndim, const int32_t * input_X_dims
  ,float * output_Y
  ,int32_t output_Y_ndim, const int32_t * output_Y_dims
  ,const char * auto_pad
  ,int32_t count_include_pad
  ,int32_t * kernel_shape
  ,int32_t number_of_kernel_shape
  ,int32_t * pads
  ,int32_t number_of_pads
  ,int32_t * strides
  ,int32_t number_of_strides
);

void ONNC_RUNTIME_batchnormalization_float(
  void * onnc_runtime_context
  ,const float * input_X
  ,int32_t input_X_ndim, const int32_t * input_X_dims
  ,const float * input_scale
  ,int32_t input_scale_ndim, const int32_t * input_scale_dims
  ,const float * input_B
  ,int32_t input_B_ndim, const int32_t * input_B_dims
  ,const float * input_mean
  ,int32_t input_mean_ndim, const int32_t * input_mean_dims
  ,const float * input_var
  ,int32_t input_var_ndim, const int32_t * input_var_dims
  ,float * output_Y
  ,int32_t output_Y_ndim, const int32_t * output_Y_dims
  ,float * output_mean
  ,int32_t output_mean_ndim, const int32_t * output_mean_dims
  ,float * output_var
  ,int32_t output_var_ndim, const int32_t * output_var_dims
  ,float * output_saved_mean
  ,int32_t output_saved_mean_ndim, const int32_t * output_saved_mean_dims
  ,float * output_saved_var
  ,int32_t output_saved_var_ndim, const int32_t * output_saved_var_dims
  ,float epsilon
  ,float momentum
  ,int32_t spatial
);

void ONNC_RUNTIME_concat_float(
  void * onnc_runtime_context
  ,const float * const * input_inputs
  ,int32_t input_inputs_ntensor
  ,const int32_t * input_inputs_ndim, const int32_t * const * input_inputs_dims
  ,float * output_concat_result
  ,int32_t output_concat_result_ndim, const int32_t * output_concat_result_dims
  ,int32_t axis
);

void ONNC_RUNTIME_conv_float(
  void * onnc_runtime_context
  ,const float * input_X
  ,int32_t input_X_ndim, const int32_t * input_X_dims
  ,const float * input_W
  ,int32_t input_W_ndim, const int32_t * input_W_dims
  ,const float * input_B
  ,int32_t input_B_ndim, const int32_t * input_B_dims
  ,float * output_Y
  ,int32_t output_Y_ndim, const int32_t * output_Y_dims
  ,const char * auto_pad
  ,int32_t * dilations
  ,int32_t number_of_dilations
  ,int32_t group
  ,int32_t * kernel_shape
  ,int32_t number_of_kernel_shape
  ,int32_t * pads
  ,int32_t number_of_pads
  ,int32_t * strides
  ,int32_t number_of_strides
);

void ONNC_RUNTIME_gemm_float(
  void * onnc_runtime_context
  ,const float * input_A
  ,int32_t input_A_ndim, const int32_t * input_A_dims
  ,const float * input_B
  ,int32_t input_B_ndim, const int32_t * input_B_dims
  ,const float * input_C
  ,int32_t input_C_ndim, const int32_t * input_C_dims
  ,float * output_Y
  ,int32_t output_Y_ndim, const int32_t * output_Y_dims
  ,float alpha
  ,float beta
  ,int32_t transA
  ,int32_t transB
);

void ONNC_RUNTIME_giventensorfill_float(
  void * onnc_runtime_context
  ,const float * input_shape
  ,int32_t input_shape_ndim, const int32_t * input_shape_dims
  ,float * output_X
  ,int32_t output_X_ndim, const int32_t * output_X_dims
  ,int32_t * extra_shape
  ,int32_t number_of_extra_shape
  ,int32_t input_as_shape
  ,int32_t * shape
  ,int32_t number_of_shape
  ,float * values
  ,int32_t number_of_values
);
void ONNC_RUNTIME_globalaveragepool_float(
  void * onnc_runtime_context
  ,const float * input_X
  ,int32_t input_X_ndim, const int32_t * input_X_dims
  ,float * output_Y
  ,int32_t output_Y_ndim, const int32_t * output_Y_dims
);

void ONNC_RUNTIME_lrn_float(
  void * onnc_runtime_context
  ,const float * input_X
  ,int32_t input_X_ndim, const int32_t * input_X_dims
  ,float * output_Y
  ,int32_t output_Y_ndim, const int32_t * output_Y_dims
  ,float alpha
  ,float beta
  ,float bias
  ,int32_t size
);

void ONNC_RUNTIME_maxpool_float(
  void * onnc_runtime_context
  ,const float * input_X
  ,int32_t input_X_ndim, const int32_t * input_X_dims
  ,float * output_Y
  ,int32_t output_Y_ndim, const int32_t * output_Y_dims
  ,float * output_Indices
  ,int32_t output_Indices_ndim, const int32_t * output_Indices_dims
  ,const char * auto_pad
  ,int32_t * kernel_shape
  ,int32_t number_of_kernel_shape
  ,int32_t * pads
  ,int32_t number_of_pads
  ,int32_t storage_order
  ,int32_t * strides
  ,int32_t number_of_strides
);

void ONNC_RUNTIME_mul_float(
  void * onnc_runtime_context
  ,const float * input_A
  ,int32_t input_A_ndim, const int32_t * input_A_dims
  ,const float * input_B
  ,int32_t input_B_ndim, const int32_t * input_B_dims
  ,float * output_C
  ,int32_t output_C_ndim, const int32_t * output_C_dims
);

void ONNC_RUNTIME_relu_float(
  void * onnc_runtime_context
  ,const float * input_X
  ,int32_t input_X_ndim, const int32_t * input_X_dims
  ,float * output_Y
  ,int32_t output_Y_ndim, const int32_t * output_Y_dims
);

void ONNC_RUNTIME_reshape_float(
  void * onnc_runtime_context
  ,const float * input_data
  ,int32_t input_data_ndim, const int32_t * input_data_dims
  ,const float * input_shape
  ,int32_t input_shape_ndim, const int32_t * input_shape_dims
  ,float * output_reshaped
  ,int32_t output_reshaped_ndim, const int32_t * output_reshaped_dims
);

void ONNC_RUNTIME_softmax_float(
  void * onnc_runtime_context
  ,const float * input_input
  ,int32_t input_input_ndim, const int32_t * input_input_dims
  ,float * output_output
  ,int32_t output_output_ndim, const int32_t * output_output_dims
  ,int32_t axis
);

void ONNC_RUNTIME_sum_float(
  void * onnc_runtime_context
  ,const float * const * input_data_0
  ,int32_t input_data_0_ntensor
  ,const int32_t * input_data_0_ndim, const int32_t * const * input_data_0_dims
  ,float * output_sum
  ,int32_t output_sum_ndim, const int32_t * output_sum_dims
);

void ONNC_RUNTIME_transpose_float(
  void * onnc_runtime_context
  ,const float * input_data
  ,int32_t input_data_ndim, const int32_t * input_data_dims
  ,float * output_transposed
  ,int32_t output_transposed_ndim, const int32_t * output_transposed_dims
  ,int32_t * perm
  ,int32_t number_of_perm
);

void ONNC_RUNTIME_unsqueeze_float(
  void * onnc_runtime_context
  ,const float * input_data
  ,int32_t input_data_ndim, const int32_t * input_data_dims
  ,float * output_expanded
  ,int32_t output_expanded_ndim, const int32_t * output_expanded_dims
  ,int32_t * axes
  ,int32_t number_of_axes
);
