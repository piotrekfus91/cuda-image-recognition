#include <gtest/gtest.h>
#include "cir/devenv/CudaDeviceChecker.cuh"
#include <iostream>

using namespace cir::devenv;

class CudaDeviceTest : public ::testing::Test {
public:
	CudaDeviceChecker cudaDeviceChecker;
};

void printProp(std::string name, std::string value) {
	std::cout << "| ";
	std::cout << name;
	for(int i = name.size(); i < 28; i++) std::cout << " ";
	std::cout << "| ";
	std::cout << value;
	for(int i = value.size(); i < 30; i++) std::cout << " ";
	std::cout << "|" << std::endl;
}

void printAllProps(DeviceProps& deviceProps) {
	std::cout << std::endl;
	std::cout << "=====================CUDA DEVICE PROPERTIES====================" << std::endl;
	printProp("Name", deviceProps.getName());
	printProp("Compute compatibility", deviceProps.getComputeCompatibility());
	printProp("Grid size", deviceProps.getMaxGridSize());
	printProp("Thread size", deviceProps.getMaxThreadsSize());
	printProp("Multi processors", deviceProps.getMultiProcessors());
	printProp("Threads per processor", deviceProps.getThreadsPerProcessor());
	printProp("Threads per block", deviceProps.getThreadsPerBlock());
	printProp("Concurrent kernels", deviceProps.getConcurrentKernels());
	printProp("Warp size", deviceProps.getWarpSize());
	printProp("Clock rate", deviceProps.getClockRate());
	std::cout << "=====================CUDA DEVICE PROPERTIES====================" << std::endl;
	std::cout << std::endl;
}

TEST_F(CudaDeviceTest, CanRunNativeCudaCode) {
	ASSERT_TRUE(cudaDeviceChecker.canRunNativeCudaCode());

	for(int device = 0; device < cudaDeviceChecker.getCudaDevicesCount(); device++) {
		DeviceProps deviceProps = cudaDeviceChecker.getCudaDeviceProperties(device);
		printAllProps(deviceProps);
	}
}
