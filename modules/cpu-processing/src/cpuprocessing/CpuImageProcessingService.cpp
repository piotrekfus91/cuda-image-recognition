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
