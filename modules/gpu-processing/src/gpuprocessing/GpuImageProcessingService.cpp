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

MatWrapper GpuImageProcessingService::lowPass(const MatWrapper& input, int size) {
	cv::gpu::GpuMat output;
	if(size == DEFAULT_LOW_PASS_KERNEL_SIZE) {
		cv::gpu::filter2D(input.getGpuMat(), output, -1, DEFAULT_LOW_PASS_KERNEL);
	} else {
		cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) / (float)(size*size);
		cv::gpu::filter2D(input.getGpuMat(), output, -1, kernel);
	}
	return output;
}

MatWrapper GpuImageProcessingService::highPass(const MatWrapper& input, int size) {
	cv::gpu::GpuMat output;
	cv::gpu::Laplacian(input.getGpuMat(), output, -1, size);
	return output;
}
