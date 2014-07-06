add_library(cpu-processing
src/cpuprocessing/CpuImageProcessingService.cpp 
)

target_link_libraries(cpu-processing common)

if(test)
	enable_testing()
endif()