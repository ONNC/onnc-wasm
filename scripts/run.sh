#!/bin/bash

print_usage(){
    echo "Usage: run.sh <mode> <model_dir> <model_name> <tensor_file> [output_file]"
    echo ""
    echo "Available mode:"
    echo "native"
    echo "ssvm"
    echo ""
    echo "Notice:"
    echo "<model_dir> is the directory of built model and libraries"
    echo "<model_name> is the model name without .onnx suffix"
    echo "[output_file] is optional, if not specified, it will be \"result.numpy\""
}

extract_basename_with_ext(){
    echo ${1##*/}
}

OUTPUT_FILE="result.numpy"
if [ $5 ];then
    OUTPUT_FILE=$(realpath $5)
fi
SSVM_VM=$2/bin/ssvm-qitc
CPU_LIBRARY_PATH=$2/lib
WEIGHT_FILE=$2/bin/$3.weight
EXECUTABLE=$2/bin/$3

if [ $# -lt 4 ]; then
    print_usage
    exit -1
elif [ $1 = "native" ]; then
    touch "result.numpy"
    LD_LIBRARY_PATH=$CPU_LIBRARY_PATH $EXECUTABLE $4 $WEIGHT_FILE
    if [ $OUTPUT_FILE != "result.numpy" ]; then
        mv "result.numpy" $OUTPUT_FILE
    fi
elif [ $1 = "ssvm" ]; then
    touch "result.numpy"
    LD_LIBRARY_PATH=$CPU_LIBRARY_PATH $SSVM_VM $EXECUTABLE.wasm $4 $WEIGHT_FILE
    if [ $OUTPUT_FILE != "result.numpy" ]; then
        mv "result.numpy" $OUTPUT_FILE
    fi
else
    print_usage
    exit -1
fi
