file(GLOB src_gpuprocessing_files "src/gpuprocessing/*.cpp")
file(GLOB test_gpuprocessing_files "test/gpuprocessing/*.cpp")

cuda_compile(SEGMENTATE_BASE src/gpuprocessing/segmentate_base.cu)
cuda_compile(DETECT_COLOR src/gpuprocessing/detect_color.cu)
cuda_compile(REGION_SPLITTING_SEGMENTATE src/gpuprocessing/region_splitting_segmentate.cu)
cuda_compile(UNION_FIND_SEGMENTATE src/gpuprocessing/union_find_segmentate.cu)
cuda_compile(COUNT_MOMENTS src/gpuprocessing/count_moments.cu)
cuda_compile(BLUR src/gpuprocessing/gpu_blur.cu)

add_library(gpu-processing
	${src_gpuprocessing_files}
	${SEGMENTATE_BASE}
	${DETECT_COLOR}
	${REGION_SPLITTING_SEGMENTATE}
	${UNION_FIND_SEGMENTATE}
	${COUNT_MOMENTS}
	${BLUR}
)

target_link_libraries(gpu-processing common)

if(test)
	enable_testing()
	
	add_executable(GpuProcessingTests
		${test_gpuprocessing_files}
	)
	
	target_link_libraries(GpuProcessingTests cpu-processing gpu-processing ${OpenCV_LIBS} gtest gtest_main pthread)
	
	add_test(Tests GpuProcessingTests)
endif()