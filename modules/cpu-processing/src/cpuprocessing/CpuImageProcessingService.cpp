#include "opencv2/imgproc/imgproc.hpp"
#include "cir/cpuprocessing/CpuImageProcessingService.h"

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

MatWrapper CpuImageProcessingService::bgrToHsv(const MatWrapper& input) {
	cv::Mat output;
	cv::cvtColor(input.getMat(), output, cv::COLOR_BGR2HSV);
	return output;
}

MatWrapper CpuImageProcessingService::hsvToBgr(const MatWrapper& input) {
	cv::Mat output;
	cv::cvtColor(input.getMat(), output, cv::COLOR_HSV2BGR);
	return output;
}

MatWrapper CpuImageProcessingService::detectColorHsv(const MatWrapper& input, const double minHue,
			const double maxHue, const double minSaturation, const double maxSaturation,
			const double minValue, const double maxValue) {
	return _cpuColorDetector.detectColorHsv(input, minHue, maxHue, minSaturation, maxSaturation,
			minValue, maxValue);
}
