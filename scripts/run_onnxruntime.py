import onnxruntime
import numpy
import struct
from functools import reduce
from sys import argv

def extract_tensor_data(tensor_offset, tensor_file):
    (offset, size) = tensor_offset
    tensor_file.seek(offset, 0)
    return tensor_file.read(size)

def read_tensors(path):
    tensor_file = open(path, "rb")
    magic = tensor_file.read(8)
    if(magic != b".TSR\x00\x00\x00\x00"):
        print("Wrong type of file {:s}".format(path))
        exit(-2)
    (number_of_tensors,) = struct.unpack("<Q", tensor_file.read(8))
    tensor_offsets = reduce(lambda acc, ind: acc + [struct.unpack("<QQ", tensor_file.read(16))], list(range(number_of_tensors)), [])
    return reduce(lambda acc, tensor_offset: acc + [extract_tensor_data(tensor_offset, tensor_file)], tensor_offsets, [])

if(len(argv) != 4):
    print("Usage: {:s} <model_file> <tensor_file> <output_file>".format(argv[0]))
    exit(-1)

model = onnxruntime.InferenceSession(argv[1])
input_tensors = read_tensors(argv[2])
input_info = model.get_inputs()
inputs = {}
for index in range(len(input_info)):
    inputs[input_info[index].name] = numpy.resize(numpy.frombuffer(input_tensors[index], dtype=numpy.float32), input_info[index].shape)
output = model.run(None, inputs)
numpy.savetxt(argv[3], output[0].flatten(order='C'), fmt='%.6f', delimiter='\n')