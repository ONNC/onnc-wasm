include_directories(${CMAKE_CURRENT_LIST_DIR})
add_library(onnc-wasm SHARED
    add.c
    averagepool.c
    batchnormalization.c
    concat.c
    conv.c
    gemm.c
    globalaveragepool.c
    lrn.c
    maxpool.c
    mul.c
    relu.c
    reshape.c
    softmax.c
    sum.c
    transpose.c
    unsqueeze.c
    ../benchmark.cpp
)
install(TARGETS onnc-wasm
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)