#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/cpuprocessing/CpuUnionFindSegmentator.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuUnionFindSegmentator.h"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/logger/BufferedConfigurableLogger.h"
#include "cir/common/logger/NullLogger.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/recognition/MetroRecognizor.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <string>
#include <list>

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::cpuprocessing;
using namespace cir::gpuprocessing;

void showRecognitionResults(MetroRecognizor& recognizor, ImageProcessingService* service);
void recognize(std::string filePath, MetroRecognizor& recognizor, ImageProcessingService* service);
void experiment(MetroRecognizor& recognizor, ImageProcessingService* service);

int main() {
	std::list<std::string> loggerConf;
	loggerConf.push_back("Segmentate");
	BufferedConfigurableLogger logger(loggerConf);
	GpuImageProcessingService service(logger);
	service.setSegmentator(new GpuUnionFindSegmentator);
	service.init(0, 0);
	service.setSegmentatorMinSize(30);

	MetroRecognizor recognizor(service);
	recognizor.learn(getTestFile("metro", "metro.png").c_str());

	showRecognitionResults(recognizor, &service);
//	experiment(recognizor, &service);

	logger.flushBuffer();
}

void showRecognitionResults(MetroRecognizor& recognizor, ImageProcessingService* service) {
	cv::namedWindow("result");
	recognize(getTestFile("metro", "metro.png"), recognizor, service);
	recognize(getTestFile("metro", "warszawa_metro_swietokrzyska.jpeg"), recognizor, service);
	recognize(getTestFile("metro", "metro_warszawa_450.jpeg"), recognizor, service);
	recognize(getTestFile("metro", "metro_mlociny.jpeg"), recognizor, service);
	recognize(getTestFile("metro", "metro-imielin.jpg"), recognizor, service);
	recognize(getTestFile("metro", "metro-otwarte.jpeg"), recognizor, service);
	recognize(getTestFile("metro", "Metro_stoklosy.jpg"), recognizor, service);
}

void experiment(MetroRecognizor& recognizor, ImageProcessingService* service) {
	int reps = 100;
	for(int i = 0; i < reps; i++) {
		recognize(getTestFile("metro", "metro.png"), recognizor, service);
	}
	for(int i = 0; i < reps; i++) {
		recognize(getTestFile("metro", "metro_mlociny.jpeg"), recognizor, service);
	}
	for(int i = 0; i < reps; i++) {
		recognize(getTestFile("metro", "metro-imielin.jpg"), recognizor, service);
	}
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
