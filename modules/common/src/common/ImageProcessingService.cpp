#include "cir/common/ImageProcessingService.h"
#include "opencv2/opencv.hpp"
#include <ctime>
#include <iostream>

using namespace cir::common;
using namespace cir::common::logger;

int ImageProcessingService::DEFAULT_LOW_PASS_KERNEL_SIZE = 3;
cv::Mat ImageProcessingService::DEFAULT_LOW_PASS_KERNEL =
		cv::Mat::ones(DEFAULT_LOW_PASS_KERNEL_SIZE, DEFAULT_LOW_PASS_KERNEL_SIZE, CV_32F)
				/ (float)(DEFAULT_LOW_PASS_KERNEL_SIZE*DEFAULT_LOW_PASS_KERNEL_SIZE);

cv::Mat ImageProcessingService::DEFAULT_ERODE_KERNEL = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));

cv::Mat ImageProcessingService::DEFAULT_DILATE_KERNEL = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));

ImageProcessingService::ImageProcessingService(Logger& logger) : _logger(logger) {

}

ImageProcessingService::~ImageProcessingService() {

}

cir::common::MatWrapper ImageProcessingService::toGrey(const cir::common::MatWrapper& input) {
	clock_t start = clock();
	cir::common::MatWrapper mw = doToGrey(input);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("To grey", elapsed_secs);
	return mw;
}

cir::common::MatWrapper ImageProcessingService::threshold(const cir::common::MatWrapper& input, double thresholdValue) {
	clock_t start = clock();
	cir::common::MatWrapper mw = doThreshold(input, thresholdValue);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("Threshold", elapsed_secs);
	return mw;
}

cir::common::MatWrapper ImageProcessingService::lowPass(const cir::common::MatWrapper& input, int size) {
	clock_t start = clock();
	cir::common::MatWrapper mw = doLowPass(input, size);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("Low pass", elapsed_secs);
	return mw;
}

cir::common::MatWrapper ImageProcessingService::highPass(const cir::common::MatWrapper& input, int size) {
	clock_t start = clock();
	cir::common::MatWrapper mw = doHighPass(input, size);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("High pass", elapsed_secs);
	return mw;
}

cir::common::MatWrapper ImageProcessingService::bgrToHsv(const cir::common::MatWrapper& input) {
	clock_t start = clock();
	cir::common::MatWrapper mw = doBgrToHsv(input);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("BGR to HSV", elapsed_secs);
	return mw;
}

cir::common::MatWrapper ImageProcessingService::hsvToBgr(const cir::common::MatWrapper& input) {
	clock_t start = clock();
	cir::common::MatWrapper mw = doHsvToBgr(input);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("HSV to BGR", elapsed_secs);
	return mw;
}

cir::common::MatWrapper ImageProcessingService::detectColorHsv(const MatWrapper& input,
		const int hsvRangesNumber, const HsvRange* hsvRanges) {
	clock_t start = clock();
	cir::common::MatWrapper mw = doDetectColorHsv(input, hsvRangesNumber, hsvRanges);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("Detect color HSV", elapsed_secs);
	return mw;
}

cir::common::MatWrapper ImageProcessingService::erode(const MatWrapper& input, int times) {
	clock_t start = clock();
	cir::common::MatWrapper mw = doErode(input, times);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("Erode", elapsed_secs);
	return mw;
}

cir::common::MatWrapper ImageProcessingService::dilate(const MatWrapper& input, int times) {
	clock_t start = clock();
	cir::common::MatWrapper mw = doDilate(input, times);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("Erode", elapsed_secs);
	return mw;
}

cir::common::SegmentArray* ImageProcessingService::segmentate(const cir::common::MatWrapper& input) {
	clock_t start = clock();
	cir::common::SegmentArray* segmentArray = doSegmentate(input);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("Segmentate", elapsed_secs);
	return segmentArray;
}

double* ImageProcessingService::countHuMoments(const cir::common::MatWrapper& input) {
	clock_t start = clock();
	double* huMoments = doCountHuMoments(input);
	clock_t stop = clock();
	double elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
	_logger.log("Count Hu moments", elapsed_secs);
	return huMoments;
}
