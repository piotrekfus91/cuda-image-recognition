file(GLOB src_cpuprocessing_files "src/cpuprocessing/*.cpp")

add_library(cpu-processing
	${src_cpuprocessing_files}
)

target_link_libraries(cpu-processing common)

if(test)
	enable_testing()
endif()