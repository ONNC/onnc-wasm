include_directories(${CMAKE_CURRENT_LIST_DIR})
find_package(JPEG REQUIRED)
add_executable(detection
    detection.cpp
)
target_link_libraries(detection
    ${JPEG_LIBRARIES}
    m
)

install(TARGETS detection
    RUNTIME
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)