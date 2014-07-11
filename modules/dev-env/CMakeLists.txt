find_package(CUDA)
cuda_compile(CUDA_DEVICE_CHECKER src/devenv/CudaDeviceChecker.cu)

if(test)
	enable_testing()
	
	cuda_add_executable(
		DevEnvTests
		src/devenv/CudaDeviceChecker.cu
		test/devenv/CudaDeviceTest.cpp
		src/devenv/OpenCvGpuChecker.cpp
		test/devenv/OpenCvGpuDeviceTest.cpp
	)
	
	target_link_libraries(DevEnvTests gtest gtest_main pthread)
	target_link_libraries(DevEnvTests ${OpenCV_LIBS})
	add_test(Tests DevEnvTests)
endif()