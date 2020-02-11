#include <cstdio>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <string>

#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h"

struct ONNC_RUNTIME_tensor_offset {
  uint64_t offset; /* Tensor offset */
  uint64_t size;   /* Size of tensor in bytes */
};

#ifndef ONNC_RUNTIME_TENSOR_FILE_MAGIC
#  define ONNC_RUNTIME_TENSOR_FILE_MAGIC ".TSR"
#endif

struct ONNC_RUNTIME_tensor_offset_table
{
  char                              magic[8]; /* Tensor File magic number. */
  uint64_t                          number_of_tensors;
  struct ONNC_RUNTIME_tensor_offset tensor_offsets[];
};

using namespace cimg_library;

int main(int argc, char const *argv[])
{
	if(argc < 3){
		std::cerr << "Usage: ./jpg2t <input_image> <output_tensor> [width] [height]" << std::endl;
		return -1;
	}
    CImg<float> image(argv[1]);
	std::uint64_t width = image.width(), height = image.height();
	if(argc >= 4){
		width = std::stoi(argv[3]);
		if(argc == 5){
			height = std::stoi(argv[4]);
		}
		std::cout << "Crop image to (" << width << " x " << height << ")" << std::endl;
		image = image.crop(0, 0, width - 1, height - 1);
	}

	const std::size_t tableSize = sizeof(ONNC_RUNTIME_tensor_offset_table) + sizeof(ONNC_RUNTIME_tensor_offset);
    auto * const table = reinterpret_cast<ONNC_RUNTIME_tensor_offset_table*>(std::calloc(tableSize, 1));
	std::strncpy(table->magic, ONNC_RUNTIME_TENSOR_FILE_MAGIC, sizeof(table->magic));
	table->number_of_tensors = 1;
	table->tensor_offsets[0].offset = tableSize;
	table->tensor_offsets[0].size   = image.size() * sizeof(float);

	std::ofstream tensorFile{argv[2], std::ios::binary};
	tensorFile.write(reinterpret_cast<const std::ofstream::char_type*>(table), tableSize);
	std::free(table);
	tensorFile.write((char*) image.data(), image.size() * sizeof(float));
	tensorFile.close();
	return 0;
}
