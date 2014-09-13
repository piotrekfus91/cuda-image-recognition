file(GLOB src_cnn_files "src/cnn/*.cpp")
file(GLOB src_cnn_neuron_files "src/cnn/neuron/*.cpp")
file(GLOB src_cnn_file_files "src/cnn/file/*.cpp")
file(GLOB test_cnn_files "test/cnn/*.cpp")
file(GLOB test_cnn_neuron_files "test/cnn/neuron/*.cpp")
file(GLOB test_cnn_file_files "test/cnn/file/*.cpp")

add_library(nn 
	${src_cnn_files}
	${src_cnn_neuron_files}
	${src_cnn_file_files}
)

set(CMAKE_CXX_FLAGS "-Wall -g -O0")

include_directories("${PROJECT_SOURCE_DIR}/include")

target_link_libraries(nn jsoncpp)

if(test)
	enable_testing()
	
	add_executable(CudaNeuronNetworkTests
		${test_cnn_files}
		${test_cnn_neuron_files}
		${test_cnn_file_files}
	)
	
	target_link_libraries(CudaNeuronNetworkTests gtest gtest_main pthread nn common)
	add_test(Tests CudaNeuronNetworkTests)
endif()
