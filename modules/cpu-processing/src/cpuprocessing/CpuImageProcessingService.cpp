#include "opencv2/imgproc/imgproc.hpp"
#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include <iostream>

using namespace cir::common;
using namespace cir::cpuprocessing;

CpuImageProcessingService::CpuImageProcessingService() {

}

CpuImageProcessingService::~CpuImageProcessingService() {

}

MatWrapper CpuImageProcessingService::toGrey(const MatWrapper& input) {
	cv::Mat output;
	cv::cvtColor(input.getMat(), output, CV_BGR2GRAY);
	return output;
}

MatWrapper CpuImageProcessingService::threshold(const MatWrapper& input, double thresholdValue) {
	cv::Mat output;
	cv::threshold(input.getMat(), output, thresholdValue, 255, cv::THRESH_BINARY);
	return output;
}

MatWrapper CpuImageProcessingService::lowPass(const MatWrapper& input, int size) {
	cv::Mat output;
	if(size == DEFAULT_LOW_PASS_KERNEL_SIZE) {
		cv::filter2D(input.getMat(), output, -1, DEFAULT_LOW_PASS_KERNEL);
	} else {
		cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) / (float)(size*size);
		cv::filter2D(input.getMat(), output, -1, kernel);
	}
	return output;
}

MatWrapper CpuImageProcessingService::highPass(const MatWrapper& input, int size) {
	cv::Mat output;
	cv::Laplacian(input.getMat(), output, -1, size);
	return output;
}
