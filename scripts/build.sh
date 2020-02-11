#!/bin/bash

print_usage(){
    echo "Usage: build.sh <mode> <onnx_file> [model_dir]"
    echo "Available mode:"
    echo "native"
    echo "ssvm"
    echo ""
    echo "NOTICES:"
    echo "* [model_dir] is optional, and will be suffixed with <mode>"
    echo ""
}

# Argument checking
if [ $# -lt 2 ]; then
    print_usage
    exit -1
fi

# Global variables
ROOT_DIR=$(pwd)
ONNX_FILE=$(realpath $2)
OUT_DIR="$PWD/out-$1"
if [ $3 ]; then
    OUT_DIR=$(realpath $3)
fi

WASI_SDK_ROOT=${WASI_SDK_ROOT:="/opt/wasi-sdk"}
WASM_BACKEND_ROOT="$PWD/${0%/*}/.."
MAKE_JOBS=${MAKE_JOBS:-""}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-"Release"}

# Start build code by cases
if [ $1 = "native" ]; then
    mkdir -p $OUT_DIR
    mkdir -p build-$1 && cd build-$1/
    if ! cmake -DBUILD_TARGET="native" -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DBUILD_C=ON -DONNX_FILE=$ONNX_FILE -DCMAKE_INSTALL_PREFIX=$OUT_DIR $WASM_BACKEND_ROOT ; then
        exit 2
    fi
    if ! make -j$MAKE_JOBS install ; then
        exit 3
    fi
    exit 0
elif [ $1 = "ssvm" ]; then
    mkdir -p $OUT_DIR
    mkdir -p build-$1 && cd build-$1/
    if ! cmake -DBUILD_TARGET="native" -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DBUILD_C=OFF -DONNX_FILE=$ONNX_FILE -DCMAKE_INSTALL_PREFIX=$OUT_DIR $WASM_BACKEND_ROOT ; then
        exit 2
    fi
    if ! make -j$MAKE_JOBS install ; then
        exit 3
    fi
    cd ${ROOT_DIR}/build-$1
    mkdir -p ssvm && cd ssvm
    if ! cmake -DONNC_WASM_ROOT=$OUT_DIR/lib -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$OUT_DIR $WASM_BACKEND_ROOT/ssvm ; then
        exit 4
    fi
    if ! make -j$MAKE_JOBS ; then
        exit 5
    fi
    cp tools/ssvm-qitc/ssvm-qitc $OUT_DIR/bin
    exit 0
else
    print_usage
    exit -4
fi
