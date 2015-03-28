#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuUnionFindSegmentator.h"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/logger/NullLogger.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/recognition/MetroRecognizor.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::cpuprocessing;
using namespace cir::gpuprocessing;

void recognize(std::string filePath, MetroRecognizor& recognizor, ImageProcessingService* service);

int main() {
	ImmediateConsoleLogger logger;
	GpuImageProcessingService service(logger);
	service.setSegmentator(new GpuUnionFindSegmentator);

	MetroRecognizor recognizor(service);
	recognizor.learn(getTestFile("metro", "metro.png").c_str());

	cv::namedWindow("result");
	recognize(getTestFile("metro", "metro.png"), recognizor, &service);
	recognize(getTestFile("metro", "warszawa_metro_swietokrzyska.jpeg"), recognizor, &service);
	recognize(getTestFile("metro", "metro_warszawa_450.jpeg"), recognizor, &service);
	recognize(getTestFile("metro", "metro_mlociny.jpeg"), recognizor, &service);
	recognize(getTestFile("metro", "metro-imielin.jpg"), recognizor, &service);
	recognize(getTestFile("metro", "metro-otwarte.jpeg"), recognizor, &service);
	recognize(getTestFile("metro", "Metro_stoklosy.jpg"), recognizor, &service);
}

void recognize(std::string filePath, MetroRecognizor& recognizor, ImageProcessingService* service) {
	cv::Mat mat = cv::imread(filePath.c_str());
	MatWrapper mw = service->getMatWrapper(mat);
	RecognitionInfo recognitionInfo = recognizor.recognize(mw);
	if(recognitionInfo.isSuccess()) {
		mw = service->mark(mw, recognitionInfo.getMatchedSegments());
	}
	mat = service->getMat(mw);
	cv::imshow("result", mat);
	cv::waitKey(0);
}
