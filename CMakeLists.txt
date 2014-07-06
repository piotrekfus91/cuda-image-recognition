cmake_minimum_required (VERSION 2.6)

option(test "Builds tests" ON)

project(CIR)

set(CMAKE_CXX_FLAGS "-Wall -g")

find_package(OpenCV REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/modules/common/include")
add_subdirectory(modules/common)

add_executable(CIR CIR.cpp)
target_link_libraries(CIR ${OpenCV_LIBS})

if(test)
	find_package(GTest REQUIRED)
	
	enable_testing()
	include_directories(${GTEST_INCLUDE_DIRS})
endif()