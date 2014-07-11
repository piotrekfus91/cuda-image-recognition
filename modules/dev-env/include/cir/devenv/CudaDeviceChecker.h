#ifndef CUDADEVICECHECKER_H_
#define CUDADEVICECHECKER_H_

namespace cir { namespace devenv {

class CudaDeviceChecker {
public:
	CudaDeviceChecker();
	virtual ~CudaDeviceChecker();

	bool canRunNativeCudaCode();
	int getCudaDevicesCount();
};

}}
#endif /* CUDADEVICECHECKER_H_ */
