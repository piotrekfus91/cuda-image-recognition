#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/gpumat.hpp"
#include "cir/gpuprocessing/GpuImageProcessingService.h"

using namespace cir::common;
using namespace cir::gpuprocessing;

GpuImageProcessingService::GpuImageProcessingService() {

}

GpuImageProcessingService::~GpuImageProcessingService() {

}

MatWrapper GpuImageProcessingService::toGrey(const MatWrapper& input) {
	cv::gpu::GpuMat output;
	cv::gpu::cvtColor(input.getGpuMat(), output, CV_BGR2GRAY);
	return output;
}

MatWrapper GpuImageProcessingService::threshold(const MatWrapper& input, double thresholdValue) {
	cv::gpu::GpuMat output;
	cv::gpu::threshold(input.getGpuMat(), output, thresholdValue, 255, cv::THRESH_BINARY);
	return output;
}
