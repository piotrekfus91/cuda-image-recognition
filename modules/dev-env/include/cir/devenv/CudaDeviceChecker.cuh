#ifndef CUDADEVICECHECKER_H_
#define CUDADEVICECHECKER_H_

#include "cir/devenv/DeviceProps.cuh"

namespace cir { namespace devenv {

class CudaDeviceChecker {
public:
	CudaDeviceChecker();
	virtual ~CudaDeviceChecker();

	bool canRunNativeCudaCode();
	int getCudaDevicesCount();

	DeviceProps getCudaDeviceProperties(int device);
};

}}
#endif /* CUDADEVICECHECKER_H_ */
