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

MatWrapper GpuImageProcessingService::bgrToHsv(const MatWrapper& input) {
	cv::gpu::GpuMat output;
	cv::gpu::cvtColor(input.getGpuMat(), output, cv::COLOR_BGR2HSV);
	return output;
}

MatWrapper GpuImageProcessingService::hsvToBgr(const MatWrapper& input) {
	cv::gpu::GpuMat output;
	cv::gpu::cvtColor(input.getGpuMat(), output, cv::COLOR_HSV2BGR);
	return output;
}

MatWrapper GpuImageProcessingService::detectColorHsv(const MatWrapper& input, const int hueNumber,
		const double* minHues, const double* maxHues, const double minSaturation,
		const double maxSaturation,	const double minValue, const double maxValue) {
	return _gpuColorDetector.detectColorHsv(input, hueNumber, minHues, maxHues, minSaturation,
			maxSaturation, minValue, maxValue);
}

SegmentArray* GpuImageProcessingService::segmentate(const cir::common::MatWrapper& input) {
	return NULL;
}

MatWrapper GpuImageProcessingService::mark(MatWrapper& input, cir::common::SegmentArray* segmentArray) {
	return input;
}

MatWrapper GpuImageProcessingService::crop(MatWrapper& input, Segment* segment) {
	cv::gpu::GpuMat inputMat = input.getGpuMat();
	cv::gpu::GpuMat outputMat;
	cv::Rect rect = cv::Rect(segment->leftX, segment->bottomY,
			segment->rightX - segment->leftX + 1, segment->topY - segment->bottomY + 1);
	inputMat(rect).copyTo(outputMat);
	return outputMat;
}
