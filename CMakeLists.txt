cmake_minimum_required (VERSION 2.6)

option(test "Builds tests" ON)

project(CIR)

set(CMAKE_CXX_FLAGS "-Wall -g -ansi")
set(CUDA_NVCC_FLAGS "-G;-g;-arch=sm_30")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

set(TEST_FILES_DIR "${PROJECT_SOURCE_DIR}/test_files")

configure_file (
	"${PROJECT_SOURCE_DIR}/modules/common/include/cir/common/config.h.in"
	"${PROJECT_SOURCE_DIR}/modules/common/include/cir/common/config.h"
)

include_directories("${PROJECT_SOURCE_DIR}/modules/common/include")
include_directories("${PROJECT_SOURCE_DIR}/modules/cpu-processing/include")
include_directories("${PROJECT_SOURCE_DIR}/modules/gpu-processing/include")
#include_directories("${PROJECT_SOURCE_DIR}/modules/cuda-neural-network/include")
include_directories("${PROJECT_SOURCE_DIR}/modules/dev-env/include")

add_subdirectory(modules/common)
add_subdirectory(modules/cpu-processing)
add_subdirectory(modules/gpu-processing)
#add_subdirectory(modules/cuda-neural-network)
add_subdirectory(modules/dev-env)

add_executable(CIR CIR.cpp)
add_executable(speed-limit-recognition speed-limit-recognition.cpp)
add_executable(registration-plate-recognition registration-plate-recognition.cpp)
add_executable(metro-recognition metro-recognition.cpp)

target_link_libraries(CIR tesseract)
target_link_libraries(CIR ${OpenCV_LIBS})
target_link_libraries(CIR dev-env)
target_link_libraries(CIR boost_thread boost_system)
target_link_libraries(CIR opencv_gpu)
target_link_libraries(CIR cpu-processing)
target_link_libraries(CIR gpu-processing)

target_link_libraries(speed-limit-recognition tesseract)
target_link_libraries(speed-limit-recognition ${OpenCV_LIBS})
target_link_libraries(speed-limit-recognition dev-env)
target_link_libraries(speed-limit-recognition boost_thread boost_system)
target_link_libraries(speed-limit-recognition opencv_gpu)
target_link_libraries(speed-limit-recognition cpu-processing)
target_link_libraries(speed-limit-recognition gpu-processing)

target_link_libraries(registration-plate-recognition tesseract)
target_link_libraries(registration-plate-recognition ${OpenCV_LIBS})
target_link_libraries(registration-plate-recognition dev-env)
target_link_libraries(registration-plate-recognition boost_thread boost_system)
target_link_libraries(registration-plate-recognition opencv_gpu)
target_link_libraries(registration-plate-recognition cpu-processing)
target_link_libraries(registration-plate-recognition gpu-processing)

target_link_libraries(metro-recognition tesseract)
target_link_libraries(metro-recognition ${OpenCV_LIBS})
target_link_libraries(metro-recognition dev-env)
target_link_libraries(metro-recognition boost_thread boost_system)
target_link_libraries(metro-recognition opencv_gpu)
target_link_libraries(metro-recognition cpu-processing)
target_link_libraries(metro-recognition gpu-processing)

if(test)
	find_package(GTest REQUIRED)
	
	enable_testing()
	include_directories(${GTEST_INCLUDE_DIRS})
endif()
