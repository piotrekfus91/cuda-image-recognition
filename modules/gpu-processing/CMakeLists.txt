file(GLOB src_gpuprocessing_files "src/gpuprocessing/*.cpp")

cuda_compile(DETECT_COLOR src/gpuprocessing/detect_color.cu)

add_library(gpu-processing
	${src_gpuprocessing_files}
	${DETECT_COLOR}
)

target_link_libraries(gpu-processing common)

if(test)
	enable_testing()
endif()