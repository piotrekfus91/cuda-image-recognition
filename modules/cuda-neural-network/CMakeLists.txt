cmake_minimum_required (VERSION 2.6)

option(test "Builds tests" ON)

file(GLOB src_cnn_files "src/cnn/*.cpp")
file(GLOB src_cnn_neuron_files "src/cnn/neuron/*.cpp")
file(GLOB test_cnn_files "test/cnn/*.cpp")
file(GLOB test_cnn_neuron_files "test/cnn/neuron/*.cpp")

add_library(nn 
	${src_cnn_files}
	${src_cnn_neuron_files}
)

set(CMAKE_CXX_FLAGS "-Wall -g -O0")

include_directories("${PROJECT_SOURCE_DIR}/include")

if(test)
	find_package(GTest REQUIRED)
	
	enable_testing()
	include_directories(${GTEST_INCLUDE_DIRS})
	
	add_executable(CudaNeuronNetworkTests ${test_cnn_files} ${test_cnn_neuron_files})
	target_link_libraries(CudaNeuronNetworkTests gtest gtest_main pthread nn)
	add_test(Tests CudaNeuronNetworkTests)
endif()
