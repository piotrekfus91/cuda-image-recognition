find_package(CUDA)
cuda_compile(CUDA_DEVICE_CHECKER src/devenv/CudaDeviceChecker.cu)

if(test)
	enable_testing()
	cuda_add_executable(DevEnvTests test/devenv/CudaDeviceTest.cpp src/devenv/CudaDeviceChecker.cu)
	target_link_libraries(DevEnvTests gtest gtest_main pthread)
	add_test(Tests DevEnvTests)
endif()