#include <stdint.h>
typedef int32_t ONNC_INDEX_TYPE;

#include "generic/assign.h"
#include "generic/binary.h"
#include "benchmark.h"

#define mul_func(T) mul_ ## T

#define generic_mul(T) \
static T mul_func(T) (T a, T b) \
{ \
  return a * b; \
}

generic_mul(float)
generic_mul(int8_t)

void ONNC_RUNTIME_mul_float(void* restrict context,
    const float* restrict A, int32_t Adim, const int32_t* restrict Ashape,
    const float* restrict B, int32_t Bdim, const int32_t* restrict Bshape,
    float* restrict C, int32_t Cdim, const int32_t* restrict Cshape)
{
#ifndef NDEBUG
    host_QITC_time_start("mul");
#endif // NDEBUG

    ONNC_ASSIGN(float, C, Cshape, Cdim, A, Ashape, Adim);
    ONNC_BINARY(float, C, Cshape, Cdim, B, Bshape, Bdim, mul_float);

#ifndef NDEBUG
    host_QITC_time_stop("mul", "operator mul");
    host_QITC_time_clear("mul");
#endif // NDEBUG
}

void ONNC_RUNTIME_mul_int8(void* restrict context,
    const int8_t* restrict A, int32_t Adim, const int32_t* restrict Ashape,
    const int8_t* restrict B, int32_t Bdim, const int32_t* restrict Bshape,
    int8_t* restrict C, int32_t Cdim, const int32_t* restrict Cshape)
{
#ifndef NDEBUG
    host_QITC_time_start("mul");
#endif // NDEBUG

    ONNC_ASSIGN(int8_t, C, Cshape, Cdim, A, Ashape, Adim);
    ONNC_BINARY(int8_t, C, Cshape, Cdim, B, Bshape, Bdim, mul_int8_t);

#ifndef NDEBUG
    host_QITC_time_stop("mul", "operator mul");
    host_QITC_time_clear("mul");
#endif // NDEBUG
}

