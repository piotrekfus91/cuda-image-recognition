#include "opencv2/imgproc/imgproc.hpp"
#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/common/exception/InvalidColorSchemeException.h"

using namespace cir::common;
using namespace cir::cpuprocessing;
using namespace cir::common::logger;
using namespace cir::common::exception;

CpuImageProcessingService::CpuImageProcessingService(Logger& logger) : ImageProcessingService(logger) {

}

CpuImageProcessingService::~CpuImageProcessingService() {
	_segmentator.shutdown();
}

void CpuImageProcessingService::init(int width, int height) {
	_logger.setModule(getModule());
	_segmentator.init(width, height);
}

const char* CpuImageProcessingService::getModule() {
	return "CPU";
}

MatWrapper CpuImageProcessingService::doToGrey(const MatWrapper& input) {
	cv::Mat output;

	if(input.getColorScheme() == MatWrapper::BGR) {
		cv::cvtColor(input.getMat(), output, CV_BGR2GRAY);
		MatWrapper mw(output);
		mw.setColorScheme(MatWrapper::GRAY);
		return mw;
	}

	if(input.getColorScheme() == MatWrapper::HSV) {
		cv::Mat hsvChannels[3];
		cv::split(input.getMat(), hsvChannels);
		MatWrapper mw(hsvChannels[2]);
		mw.setColorScheme(MatWrapper::GRAY);
		return mw;
	}

	return input;
}

MatWrapper CpuImageProcessingService::doThreshold(const MatWrapper& input, double thresholdValue) {
	cv::Mat output;
	cv::threshold(input.getMat(), output, thresholdValue, 255, cv::THRESH_BINARY);
	MatWrapper mw(output);
	mw.setColorScheme(MatWrapper::GRAY);
	return mw;
}

MatWrapper CpuImageProcessingService::doLowPass(const MatWrapper& input, int size) {
	cv::Mat outputMat;
	if(size == DEFAULT_LOW_PASS_KERNEL_SIZE) {
		cv::filter2D(input.getMat(), outputMat, -1, DEFAULT_LOW_PASS_KERNEL);
	} else {
		cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) / (float)(size*size);
		cv::filter2D(input.getMat(), outputMat, -1, kernel);
	}
	MatWrapper outputMw(outputMat);
	outputMw.setColorScheme(input.getColorScheme());
	return outputMw;
}

MatWrapper CpuImageProcessingService::doHighPass(const MatWrapper& input, int size) {
	cv::Mat outputMat;
	cv::Laplacian(input.getMat(), outputMat, -1, size);
	MatWrapper outputMw(outputMat);
	outputMw.setColorScheme(input.getColorScheme());
	return outputMw;
}

MatWrapper CpuImageProcessingService::doBgrToHsv(const MatWrapper& input) {
	if(input.getColorScheme() != MatWrapper::BGR)
		throw InvalidColorSchemeException();

	cv::Mat output;
	cv::cvtColor(input.getMat(), output, cv::COLOR_BGR2HSV);
	MatWrapper mw(output);
	mw.setColorScheme(MatWrapper::HSV);
	return mw;
}

MatWrapper CpuImageProcessingService::doHsvToBgr(const MatWrapper& input) {
	if(input.getColorScheme() != MatWrapper::HSV)
		throw InvalidColorSchemeException();

	cv::Mat output;
	cv::cvtColor(input.getMat(), output, cv::COLOR_HSV2BGR);
	MatWrapper outputMw(output);
	outputMw.setColorScheme(MatWrapper::BGR);
	return outputMw;
}

MatWrapper CpuImageProcessingService::doDetectColorHsv(const MatWrapper& input, const int hsvRangesNumber,
		const HsvRange* hsvRanges) {
	return _cpuColorDetector.detectColorHsv(input, hsvRangesNumber, hsvRanges);
}

SegmentArray* CpuImageProcessingService::doSegmentate(const MatWrapper& input) {
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
	MatWrapper outputMw(outputMat);
	outputMw.setColorScheme(input.getColorScheme());
	return outputMw;
}

double* CpuImageProcessingService::doCountHuMoments(const MatWrapper& matWrapper) {
	MatWrapper input = matWrapper;
	return _cpuMomentCounter.countHuMoments(input);
}
