#include <onnc-runtime.h>
#include <benchmark.h>

#include <limits.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LOOP_COUNT
#define LOOP_COUNT 1
#endif

struct ONNC_RUNTIME_tensor_file *create_tensor_file() {
  struct ONNC_RUNTIME_tensor_file *file =
    malloc(sizeof(struct ONNC_RUNTIME_tensor_file));
  file->data = NULL;
  return file;
}

int open_input_tensor_file(const char *filename,
                           struct ONNC_RUNTIME_tensor_file *filestruct) {
  FILE *const stream = fopen(filename, "rb");
  if (stream == NULL || filestruct == NULL) {
    return -1;
  }

  size_t tensor_table_size = sizeof(struct ONNC_RUNTIME_tensor_offset_table) +
                             sizeof(struct ONNC_RUNTIME_tensor_offset);

  // Read file
  fseek(stream, 0, SEEK_END);
  const long file_size = ftell(stream);
  if (filestruct->data == NULL) {
    filestruct->data = malloc(file_size + tensor_table_size);
  }

  fseek(stream, 0, SEEK_SET);
  fread(filestruct->data + tensor_table_size, 1, file_size, stream);

  fclose(stream);

  // Fill tensor_offset_table
  struct ONNC_RUNTIME_tensor_offset_table *offset_table =
      (struct ONNC_RUNTIME_tensor_offset_table *)filestruct->data;
  *(uint64_t *)offset_table->magic = 0ULL;
  offset_table->magic[0] = '.';
  offset_table->magic[1] = 'T';
  offset_table->magic[2] = 'S';
  offset_table->magic[3] = 'R';
  offset_table->number_of_tensors = 1;
  offset_table->tensor_offsets[0].offset = 32;
  offset_table->tensor_offsets[0].size = file_size;
  printf("File_size: %lld\n", offset_table->tensor_offsets[0].size);
  return 0;
}

int open_tensor_file(const char *filename,
                     struct ONNC_RUNTIME_tensor_file *filestruct) {
  FILE *const stream = fopen(filename, "rb");
  if (stream == NULL || filestruct == NULL) {
    return -1;
  }
  fseek(stream, 0, SEEK_END);
  const long file_size = ftell(stream);
  if (filestruct->data == NULL) {
    filestruct->data = malloc(file_size);
  }

  fseek(stream, 0, SEEK_SET);
  fread(filestruct->data, 1, file_size, stream);

  fclose(stream);

  return 0;
}

int close_tensor_file(struct ONNC_RUNTIME_tensor_file *file) {
  if (file == NULL) {
    return -1;
  }

  if (file->data) {
    free(file->data);
  }
  free(file);

  return 0;
}

void write_output(struct ONNC_RUNTIME_inference_context* context, struct ONNC_RUNTIME_tensor_view output) {
  int8_t *const values = output.data;
  printf("Output size: %ld\n", output.size);
  FILE *resultFile = fopen("result.numpy", "wb");
  fwrite(values, 1, output.size, resultFile);
  fclose(resultFile);
}

void write_output_debug(struct ONNC_RUNTIME_inference_context* context, struct ONNC_RUNTIME_tensor_view output) {
  int8_t *const values = output.data;
  const size_t count = output.size / sizeof(int8_t);
  fprintf(stderr, "[");
  for (size_t idx = 0; idx < count; ++idx) {
    fprintf(stderr, "%f, ", values[idx] * context->output_scaling_factor);
  }
  fprintf(stderr, "]");
  write_output(context, output);
}

int main(int argc, char *argv[]) {
  // Read weight
  if (argc < 3) {
    fprintf(stderr, "usage: %s <foo.tensor> <foo.weight>\n", argv[0]);
    return EXIT_FAILURE;
  }
  struct ONNC_RUNTIME_tensor_file *weight = create_tensor_file();
  struct ONNC_RUNTIME_tensor_file *input = create_tensor_file();
  if (open_tensor_file(argv[2], weight) != 0) {
    return -1;
  }

  // Prepate FIFOs
  const size_t activation_size = ONNC_RUNTIME_get_activation_memory_size();
  void *activation = malloc(activation_size);

#ifdef NDEBUG
  struct ONNC_RUNTIME_inference_context context = {.weight = weight,
                                                   .id = 0,
                                                   .completed = write_output,
                                                   .activation = activation};
#else
  struct ONNC_RUNTIME_inference_context context = {
      .weight = weight, .id = 0, .completed = write_output_debug, .activation = activation};
#endif
  // Main loop
  for (int i = 0; i < LOOP_COUNT; ++i) { // TODO: infinite loop
    if (open_tensor_file(argv[1], input) < 0) {
      printf("Open %s\n file failure", argv[1]);
      return -1;
    }
    context.input = input;
    QITC_time_start();
    ONNC_RUNTIME_model_main(&context);
    QITC_time_stop();
    QITC_time_clear();
  }
  close_tensor_file(weight);
  close_tensor_file(input);
  return EXIT_SUCCESS;
}
