cmake_minimum_required(VERSION 3.2.3)
project(onnc-wasm-backend)
set(PROJECT_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(CMAKE_C_FLAGS "$ENV{CFLAGS} -Wno-attributes")
SET(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -Wall -g")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wall -g")
set(BUILD_TARGET "rb3" CACHE STRING "Target of onnc implementation")
set_property(CACHE BUILD_TARGET PROPERTY STRINGS rb3 native node)
set(BUILD_C OFF CACHE BOOL "Build native C instead of Wasm")
set(BUILD_LIBS ON CACHE BOOL "Build libraries")
set(BUILD_ONNX ON CACHE BOOL "Build binary of ONNX model")
set(BUILD_TOOLS ON CACHE BOOL "Build tools")

if(BUILD_TOOLS)
    add_subdirectory(tools)
endif(BUILD_TOOLS)

if(BUILD_C)
else(BUILD_C)
    set(WASI_SDK_ROOT "/opt/wasi-sdk" CACHE PATH "Path of wasi-sdk")
endif(BUILD_C)

if(BUILD_LIBS)
    add_subdirectory(onnc-runtime)
    add_subdirectory(onnc-wasm)
    set(ONNC_RUNTIME_LIBS onnc-runtime onnc-wasm)
    set(ONNC_RUNTIME_LIB_STATIC onnc-runtime)
else(BUILD_LIBS)
    find_package(ONNCRuntimeLibs REQUIRED)
endif(BUILD_LIBS)

if(BUILD_ONNX)
    set(ONNX_FILE "${CMAKE_CURRENT_LIST_DIR}/model.onnx" CACHE FILEPATH "ONNX model file")
    get_filename_component(Model_name ${ONNX_FILE} NAME_WE)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_LIST_DIR}/tools/onnc
        COMMAND tar -zxvf ${CMAKE_CURRENT_LIST_DIR}/tools/onnc.tar.gz -C ${CMAKE_CURRENT_LIST_DIR}/tools
    )
    add_custom_command(OUTPUT ${Model_name}.c
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/tools/onnc
        COMMAND ${CMAKE_CURRENT_LIST_DIR}/tools/onnc -mquadruple C -o ${CMAKE_CURRENT_BINARY_DIR}/${Model_name}.c ${ONNX_FILE}
    )
    if(BUILD_C)
        set(EXENAME ${Model_name})
    else(BUILD_C)
        set(CMAKE_C_COMPILER "${WASI_SDK_ROOT}/bin/clang")
        set(CMAKE_CXX_COMPILER "${WASI_SDK_ROOT}/bin/clang++")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=wasm32-wasi --sysroot=${WASI_SDK_ROOT}/share/wasi-sysroot -Wl,--allow-undefined-file=${CMAKE_CURRENT_LIST_DIR}/onnc-runtime/import_funcs.txt")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --target=wasm32-wasi --sysroot=${WASI_SDK_ROOT}/share/wasi-sysroot -Wl,--allow-undefined-file=${CMAKE_CURRENT_LIST_DIR}/onnc-runtime/import_funcs.txt")
        set(CMAKE_AR "${WASI_SDK_ROOT}/bin/llvm-ar")
        set(CMAKE_RANLIB "${WASI_SDK_ROOT}/bin/llvm-ranlib")
        set(EXENAME ${Model_name}.wasm)
    endif(BUILD_C)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/onnc-runtime/include)
    add_executable(${EXENAME}
        ${Model_name}.c
    )

    if(BUILD_C)
        target_link_libraries(${EXENAME}
            ${ONNC_RUNTIME_LIBS}
        )
    else(BUILD_C)
        target_link_libraries(${EXENAME}
            ${ONNC_RUNTIME_LIB_STATIC}
        )
    endif(BUILD_C)

    install(TARGETS ${EXENAME}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${Model_name}.weight
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    )
endif(BUILD_ONNX)
