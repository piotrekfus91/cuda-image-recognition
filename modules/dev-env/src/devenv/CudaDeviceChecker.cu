#include <iostream>

#include "cir/devenv/CudaDeviceChecker.h"

using namespace cir::devenv;

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
