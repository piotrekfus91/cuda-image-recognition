file(GLOB src_cpuprocessing_files "src/cpuprocessing/*.cpp")
file(GLOB test_cpuprocessing_files "test/cpuprocessing/*.cpp")

add_library(cpu-processing
	${src_cpuprocessing_files}
)

target_link_libraries(cpu-processing common)

if(test)
	enable_testing()
	
	add_executable(CpuProcessingTests
		${test_cpuprocessing_files}
	)
	
	target_link_libraries(CpuProcessingTests cpu-processing ${OpenCV_LIBS} gtest gtest_main pthread)
	
	add_test(Tests CpuProcessingTests)
endif()