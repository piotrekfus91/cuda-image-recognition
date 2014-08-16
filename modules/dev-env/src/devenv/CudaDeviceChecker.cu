#include "cir/devenv/CudaDeviceChecker.cuh"

namespace cir { namespace devenv {

CudaDeviceChecker::CudaDeviceChecker() {

}

CudaDeviceChecker::~CudaDeviceChecker() {

}

bool CudaDeviceChecker::canRunNativeCudaCode() {
	return getCudaDevicesCount() > 0;
}

int CudaDeviceChecker::getCudaDevicesCount() {
	int count;
	cudaGetDeviceCount(&count);
	return count;
}

DeviceProps CudaDeviceChecker::getCudaDeviceProperties(int device) {
	DeviceProps deviceProps(device);
	return deviceProps;
}

}}
