find_library(ONNC_runtime_lib NAMES libonnc-runtime.so
    PATHS ${PROJECT_ROOT}/out/lib
)
find_library(ONNC_runtime_lib_static NAMES libonnc-runtime.a
    PATHS ${PROJECT_ROOT}/out/lib
)
find_library(ONNC_wasm_lib NAMES libonnc-wasm.so
    PATHS ${PROJECT_ROOT}/out/lib
)

set(ONNC_RUNTIME_LIBS ${ONNC_runtime_lib} ${ONNC_wasm_lib})
set(ONNC_RUNTIME_LIB_STATIC ${ONNC_runtime_lib_static})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNCRuntimeLibs DEFAULT_MSG
    ONNC_RUNTIME_LIBS
    ONNC_RUNTIME_LIB_STATIC
)