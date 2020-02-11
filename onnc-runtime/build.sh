#!/bin/bash

# Native C
mkdir -p build-c && cd build-c && cmake -DBUILD_WASM=OFF .. && make install \
    && cd .. && rm -rf build-c

# Wasm
export CC=${CC_WASM:-clang}
export CXX=${CXX_WASM:-clang++}
export AR=${AR_WASM:-llvm-ar}
export RANLIB=${RANLIB_WASM:-llvm-ranlib}
mkdir -p build-wasm && cd build-wasm && cmake .. && make install \
    && cd .. && rm -rf build-wasm