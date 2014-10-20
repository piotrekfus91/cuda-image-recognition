file(GLOB src_common_files "src/common/*.cpp")
file(GLOB src_common_exception_files "src/common/exception/*.cpp")
file(GLOB src_common_logger_files "src/common/logger/*.cpp")
file(GLOB src_common_recognition_files "src/common/recognition/*.cpp")
file(GLOB test_common_files "test/common/*.cpp")

cuda_compile(CUDA_UTIL src/common/cuda_host_util.cu)

add_library(common
	${src_common_files}
	${src_common_exception_files}
	${src_common_logger_files}
	${src_common_recognition_files}
	${CUDA_UTIL}
)

if(test)
	enable_testing()
	
	add_executable(CommonTests
		${test_common_files}
	)
	
	target_link_libraries(CommonTests gtest gtest_main pthread)
	target_link_libraries(CommonTests ${OpenCV_LIBS})
	target_link_libraries(CommonTests common)
	
	add_test(Tests CommonTests)
endif()
