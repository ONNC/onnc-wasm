import numpy
from sys import argv

if(len(argv) != 3):
    print("Usage: {:s} <raw_output_file> <text_output_file>".format(argv[0]))
    exit(-1)

numpy_file = open(argv[1], "rb")
numpy_array = numpy.frombuffer(numpy_file.read(), dtype=numpy.float32)
print(len(numpy_array))
numpy.savetxt(argv[2], numpy_array, fmt='%.6f')