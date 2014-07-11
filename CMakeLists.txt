cmake_minimum_required (VERSION 2.6)

option(test "Builds tests" ON)

project(CIR)

set(CMAKE_CXX_FLAGS "-Wall -g")

find_package(OpenCV REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/modules/common/include")
include_directories("${PROJECT_SOURCE_DIR}/modules/cpu-processing/include")
include_directories("${PROJECT_SOURCE_DIR}/modules/gpu-processing/include")
include_directories("${PROJECT_SOURCE_DIR}/modules/dev-env/include")

add_subdirectory(modules/common)
add_subdirectory(modules/cpu-processing)
add_subdirectory(modules/gpu-processing)
add_subdirectory(modules/dev-env)

add_executable(CIR CIR.cpp)

target_link_libraries(CIR ${OpenCV_LIBS})
target_link_libraries(CIR opencv_gpu)
target_link_libraries(CIR cpu-processing)
target_link_libraries(CIR gpu-processing)

if(test)
	find_package(GTest REQUIRED)
	
	enable_testing()
	include_directories(${GTEST_INCLUDE_DIRS})
endif()