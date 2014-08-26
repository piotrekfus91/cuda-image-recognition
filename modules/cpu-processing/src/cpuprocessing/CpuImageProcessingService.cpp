#include "opencv2/imgproc/imgproc.hpp"
#include "cir/cpuprocessing/CpuImageProcessingService.h"

using namespace cir::common;
using namespace cir::cpuprocessing;

CpuImageProcessingService::CpuImageProcessingService() {

}

CpuImageProcessingService::~CpuImageProcessingService() {
	_segmentator.shutdown();
}

void CpuImageProcessingService::init(int width, int height) {
	_segmentator.init(width, height);
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

MatWrapper CpuImageProcessingService::detectColorHsv(const MatWrapper& input, const int hueNumber,
		const double* minHues, const double* maxHues, const double minSaturation, const double maxSaturation,
		const double minValue, const double maxValue) {
	return _cpuColorDetector.detectColorHsv(input, hueNumber, minHues, maxHues, minSaturation, maxSaturation,
			minValue, maxValue);
}

SegmentArray* CpuImageProcessingService::segmentate(const MatWrapper& input) {
	return _segmentator.segmentate(input);
}

MatWrapper CpuImageProcessingService::mark(MatWrapper& input, SegmentArray* segmentArray) {
	return _marker.markSegments(input, segmentArray);
}

MatWrapper CpuImageProcessingService::crop(MatWrapper& input, Segment* segment) {
	cv::Mat inputMat = input.getMat();
	cv::Mat outputMat;
	cv::Rect rect = cv::Rect(segment->leftX, segment->bottomY,
			segment->rightX - segment->leftX + 1, segment->topY - segment->bottomY + 1);
	inputMat(rect).copyTo(outputMat);
	return outputMat;
}

double* CpuImageProcessingService::countHuMoments(const MatWrapper& matWrapper) {
	MatWrapper input = matWrapper;
	return _cpuMomentCounter.countHuMoments(input);
}
