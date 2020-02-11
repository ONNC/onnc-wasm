#include <stdint.h>
typedef int32_t ONNC_INDEX_TYPE;

#include "generic/assign.h"
#include "generic/binary.h"
#include "benchmark.h"

#define add_func(T) add_ ## T

#define generic_add(T) \
static T add_func(T) (T a, T b) \
{ \
  return a + b; \
}

generic_add(float)
generic_add(int8_t)

void ONNC_RUNTIME_add_float(void* restrict context,
    const float* restrict A, int32_t Adim, const int32_t* restrict Ashape,
    const float* restrict B, int32_t Bdim, const int32_t* restrict Bshape,
    float* restrict C, int32_t Cdim, const int32_t* restrict Cshape)
{
#ifndef NDEBUG
    host_QITC_time_start("add");
#endif // NDEBUG

    ONNC_ASSIGN(float, C, Cshape, Cdim, A, Ashape, Adim);
    ONNC_BINARY(float, C, Cshape, Cdim, B, Bshape, Bdim, add_func(float));

#ifndef NDEBUG
    host_QITC_time_stop("add", "operator add");
    host_QITC_time_clear("add");
#endif // NDEBUG
}

// TODO: Quantization
void ONNC_RUNTIME_add_int8(void * restrict onnc_runtime_context
  ,const int8_t * restrict A ,int32_t Adim, const int32_t * restrict Ashape
  ,const int8_t * restrict B ,int32_t Bdim, const int32_t * restrict Bshape
  ,int8_t * restrict C ,int32_t Cdim, const int32_t * restrict Cshape
){
#ifndef NDEBUG
    host_QITC_time_start("add");
#endif // NDEBUG

    ONNC_ASSIGN(int8_t, C, Cshape, Cdim, A, Ashape, Adim);
    ONNC_BINARY(int8_t, C, Cshape, Cdim, B, Bshape, Bdim, add_func(int8_t));

#ifndef NDEBUG
    host_QITC_time_stop("add", "operator add");
    host_QITC_time_clear("add");
#endif // NDEBUG
}
