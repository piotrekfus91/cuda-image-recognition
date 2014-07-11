add_library(gpu-processing
src/gpuprocessing/GpuImageProcessingService.cpp 
)

target_link_libraries(gpu-processing common)

if(test)
	enable_testing()
endif()