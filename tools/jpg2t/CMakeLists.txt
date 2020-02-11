find_package(JPEG REQUIRED)
add_executable(jpg2t
    jpg2t.cpp
)
target_link_libraries(jpg2t
    pthread
    ${JPEG_LIBRARIES}
)
install(TARGETS jpg2t
    RUNTIME
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)