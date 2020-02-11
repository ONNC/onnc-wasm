include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB_RECURSE OPERATOR_C_FILES lib/operator/*.c)

if(BUILD_C)
    add_library(onnc-runtime SHARED
        ${OPERATOR_C_FILES}
        lib/onnc-runtime.c
        "export/client-app.c"
        "export/client-lib.c"
        "export/onnc-runtime-core.c"
        lib/benchmark.cpp
	)
    target_link_libraries(onnc-runtime
        m
    )
else(BUILD_C)
    set(CMAKE_C_COMPILER "${WASI_SDK_ROOT}/bin/clang")
    set(CMAKE_CXX_COMPILER "${WASI_SDK_ROOT}/bin/clang++")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=wasm32-wasi --sysroot=${WASI_SDK_ROOT}/share/wasi-sysroot")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --target=wasm32-wasi --sysroot=${WASI_SDK_ROOT}/share/wasi-sysroot")
    set(CMAKE_AR "${WASI_SDK_ROOT}/bin/llvm-ar")
    set(CMAKE_RANLIB "${WASI_SDK_ROOT}/bin/llvm-ranlib")
    add_library(onnc-runtime STATIC
        ${OPERATOR_C_FILES}
        lib/onnc-runtime.c
        "export/client-app.c"
        "export/client-lib.c"
        "export/onnc-runtime-core.c"
    )
endif(BUILD_C)

install(TARGETS onnc-runtime
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
