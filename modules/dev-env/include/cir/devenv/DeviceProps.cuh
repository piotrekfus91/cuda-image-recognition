#ifndef DEVICEPROPS_H_
#define DEVICEPROPS_H_

#include <vector_types.h>
#include <string>

namespace cir { namespace devenv {

class DeviceProps {
public:
	DeviceProps(int device);
	virtual ~DeviceProps();

	std::string getName();
	std::string getComputeCompatibility();
	std::string getMaxGridSize();
	std::string getMaxThreadsSize();
	std::string getMultiProcessors();
	std::string getConcurrentKernels();
	std::string getClockRate();
	std::string getThreadsPerProcessor();

private:
	int _device;
	cudaDeviceProp _deviceProp;
};

}}
#endif
