file(GLOB src_gpuprocessing_files "src/gpuprocessing/*.cpp")

add_library(gpu-processing
	${src_gpuprocessing_files}
)

target_link_libraries(gpu-processing common)

if(test)
	
endif()