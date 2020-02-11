#include <stdio.h>
#include <stdint.h>
#include <onnc-runtime.h>

void ONNC_RUNTIME_print_per_layer_int8(
    const char* name,
    const int8_t* tensor,
    const int32_t ndim,
    const int32_t* dims,
    const float scaling_factor
){
    FILE* fout = fopen(name, "wb");
    FILE* flist = fopen("list.txt", "a");
    fprintf(flist, "%s\n", name);
    fclose(flist);
    int32_t size = 1;
    for(int32_t i = 0; i < ndim; ++i){
        size *= dims[i];
    }
    for(int32_t i = 0; i < size; ++i){
        float value = tensor[i] * scaling_factor;
        fwrite(&value, sizeof(float), 1, fout);
    }
    fclose(fout);
}

void ONNC_RUNTIME_print_per_layer_float(
    const char* name,
    const float* tensor,
    const int32_t ndim,
    const int32_t* dims
){
    FILE* fout = fopen(name, "wb");
    FILE* flist = fopen("list.txt", "a");
    fprintf(flist, "%s\n", name);
    fclose(flist);
    int32_t size = 1;
    for(int32_t i = 0; i < ndim; ++i){
        size *= dims[i];
    }
    fwrite(tensor, sizeof(float), size, fout);
    fclose(fout);
}