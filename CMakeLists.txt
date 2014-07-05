cmake_minimum_required (VERSION 2.6)
project(CIR)

find_package(OpenCV REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory(modules/common)

add_executable(CIR CIR.cpp)
target_link_libraries(CIR ${OpenCV_LIBS})
