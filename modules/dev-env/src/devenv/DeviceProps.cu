#include "cir/devenv/DeviceProps.cuh"
#include <sstream>

namespace cir { namespace devenv {

DeviceProps::DeviceProps(int device) : _device(device) {
	cudaGetDeviceProperties(&_deviceProp, _device);
}

DeviceProps::~DeviceProps() {

}

std::string DeviceProps::getName() {
	return _deviceProp.name;
}

std::string DeviceProps::getComputeCompatibility() {
	std::ostringstream str;
	str <<_deviceProp.major;
	str << ".";
	str << _deviceProp.minor;
	return str.str();
}

std::string DeviceProps::getMaxGridSize() {
	std::ostringstream str;
	str << _deviceProp.maxGridSize[0];
	str << ", ";
	str << _deviceProp.maxGridSize[1];
	str << ", ";
	str << _deviceProp.maxGridSize[2];
	return str.str();
}

std::string DeviceProps::getMaxThreadsSize() {
	std::ostringstream str;
	str << _deviceProp.maxThreadsDim[0];
	str << ", ";
	str << _deviceProp.maxThreadsDim[1];
	str << ", ";
	str << _deviceProp.maxThreadsDim[2];
	return str.str();
}

std::string DeviceProps::getMultiProcessors() {
	std::ostringstream str;
	str << _deviceProp.multiProcessorCount;
	return str.str();
}

std::string DeviceProps::getConcurrentKernels() {
	std::ostringstream str;
	str << _deviceProp.concurrentKernels;
	return str.str();
}

std::string DeviceProps::getClockRate() {
	std::ostringstream str;
	str << _deviceProp.clockRate;
	return str.str();
}

std::string DeviceProps::getThreadsPerProcessor() {
	std::ostringstream str;
	str << _deviceProp.maxThreadsPerMultiProcessor;
	return str.str();
}

std::string DeviceProps::getThreadsPerBlock() {
	std::ostringstream str;
	str << _deviceProp.maxThreadsPerBlock;
	return str.str();
}

}}
