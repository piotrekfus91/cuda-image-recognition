#include <iostream>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/gpumat.hpp"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/common/exception/InvalidColorSchemeException.h"
#include "cir/common/cuda_host_util.cuh"

using namespace cir::common;
using namespace cir::gpuprocessing;
using namespace cir::common::logger;
using namespace cir::common::exception;

GpuImageProcessingService::GpuImageProcessingService(cir::common::logger::Logger& logger) : ImageProcessingService(logger) {

}

GpuImageProcessingService::~GpuImageProcessingService() {

}

void GpuImageProcessingService::init(int width, int height) {
	_logger.setModule(getModule());
	_segmentator.init(width, height);
	_gpuMomentCounter.init(width, height);
	Logger* kernelLogger = _logger.clone();
	kernelLogger->setModule("Kernel");
	set_default_logger(kernelLogger);
}

const char* GpuImageProcessingService::getModule() {
	return "GPU";
}

MatWrapper GpuImageProcessingService::doToGrey(const MatWrapper& input) {
	cv::gpu::GpuMat output;

	if(input.getColorScheme() == MatWrapper::BGR) {
		cv::cvtColor(input.getGpuMat(), output, CV_BGR2GRAY);
		MatWrapper mw(output);
		mw.setColorScheme(MatWrapper::GRAY);
		return mw;
	}

	if(input.getColorScheme() == MatWrapper::HSV) {
		cv::gpu::GpuMat hsvChannels[3];
		cv::gpu::split(input.getGpuMat(), hsvChannels);
		MatWrapper mw(hsvChannels[2]);
		mw.setColorScheme(MatWrapper::GRAY);
		return mw;
	}

	return input;
}

MatWrapper GpuImageProcessingService::doThreshold(const MatWrapper& input, double thresholdValue) {
	cv::gpu::GpuMat output;
	cv::gpu::threshold(input.getGpuMat(), output, thresholdValue, 255, cv::THRESH_BINARY);
	MatWrapper mw(output);
	mw.setColorScheme(MatWrapper::GRAY);
	return mw;
}

MatWrapper GpuImageProcessingService::doLowPass(const MatWrapper& input, int size) {
	cv::gpu::GpuMat output;
	if(size == DEFAULT_LOW_PASS_KERNEL_SIZE) {
		cv::gpu::filter2D(input.getGpuMat(), output, -1, DEFAULT_LOW_PASS_KERNEL);
	} else {
		cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) / (float)(size*size);
		cv::gpu::filter2D(input.getGpuMat(), output, -1, kernel);
	}

	MatWrapper mw(output);
	mw.setColorScheme(input.getColorScheme());
	return mw;
}

MatWrapper GpuImageProcessingService::doHighPass(const MatWrapper& input, int size) {
	cv::gpu::GpuMat output;
	cv::gpu::Laplacian(input.getGpuMat(), output, -1, size);
	MatWrapper mw(output);
	mw.setColorScheme(input.getColorScheme());
	return output;
}

MatWrapper GpuImageProcessingService::doBgrToHsv(const MatWrapper& input) {
	if(input.getColorScheme() != MatWrapper::BGR)
		throw InvalidColorSchemeException();

	cv::gpu::GpuMat output;
	cv::gpu::cvtColor(input.getGpuMat(), output, cv::COLOR_BGR2HSV);
	MatWrapper mw(output);
	mw.setColorScheme(MatWrapper::HSV);
	return mw;
}

MatWrapper GpuImageProcessingService::doHsvToBgr(const MatWrapper& input) {
	if(input.getColorScheme() != MatWrapper::HSV)
		throw InvalidColorSchemeException();

	cv::gpu::GpuMat output;
	cv::gpu::cvtColor(input.getGpuMat(), output, cv::COLOR_HSV2BGR);
	MatWrapper mw(output);
	mw.setColorScheme(MatWrapper::BGR);
	return output;
}

MatWrapper GpuImageProcessingService::doDetectColorHsv(const MatWrapper& input, const int hsvRangesNumber,
		const HsvRange* hsvRanges) {
	return _gpuColorDetector.detectColorHsv(input, hsvRangesNumber, hsvRanges);
}

SegmentArray* GpuImageProcessingService::doSegmentate(const cir::common::MatWrapper& input) {
	cir::common::MatWrapper matWrapper = input;
	return _segmentator.segmentate(matWrapper);
}

MatWrapper GpuImageProcessingService::mark(MatWrapper& input, cir::common::SegmentArray* segmentArray) {
	return input; // TODO
}

MatWrapper GpuImageProcessingService::crop(MatWrapper& input, Segment* segment) {
	cv::gpu::GpuMat inputMat = input.getGpuMat();
	cv::gpu::GpuMat outputMat;
	cv::Rect rect = cv::Rect(segment->leftX, segment->bottomY,
			segment->rightX - segment->leftX + 1, segment->topY - segment->bottomY + 1);
	inputMat(rect).copyTo(outputMat);
	MatWrapper mw(outputMat);
	mw.setColorScheme(input.getColorScheme());
	return mw;
}

double* GpuImageProcessingService::doCountHuMoments(const MatWrapper& matWrapper) {
	MatWrapper input = matWrapper;
	return _gpuMomentCounter.countHuMoments(input);
}
