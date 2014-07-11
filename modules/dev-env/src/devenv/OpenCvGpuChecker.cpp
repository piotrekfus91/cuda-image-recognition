#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "cir/devenv/OpenCvGpuChecker.h"

using namespace cir::devenv;

OpenCvGpuChecker::OpenCvGpuChecker() {

}

OpenCvGpuChecker::~OpenCvGpuChecker() {

}

bool OpenCvGpuChecker::canRunOpenCvGpuCode() {
	return getOpenCvGpuDeviceCount() > 0;
}

int OpenCvGpuChecker::getOpenCvGpuDeviceCount() {
	return cv::gpu::getCudaEnabledDeviceCount();
}
