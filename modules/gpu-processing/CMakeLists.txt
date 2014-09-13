file(GLOB src_gpuprocessing_files "src/gpuprocessing/*.cpp")
file(GLOB test_gpuprocessing_files "test/gpuprocessing/*.cpp")

cuda_compile(DETECT_COLOR src/gpuprocessing/detect_color.cu)
cuda_compile(REGION_SPLITTING_SEGMENTATE src/gpuprocessing/region_splitting_segmentate.cu)

add_library(gpu-processing
	${src_gpuprocessing_files}
	${DETECT_COLOR}
	${REGION_SPLITTING_SEGMENTATE}
)

target_link_libraries(gpu-processing common)

if(test)
	enable_testing()
	
	add_executable(GpuProcessingTests
		${src_gpuprocessing_files}
		${test_gpuprocessing_files}
	)
	
	target_link_libraries(GpuProcessingTests gtest gtest_main pthread ${OpenCV_LIBS} gpu-processing)
	
	add_test(Tests GpuProcessingTests)
endif()