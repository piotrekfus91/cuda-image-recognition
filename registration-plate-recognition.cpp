#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/cpuprocessing/CpuUnionFindSegmentator.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuUnionFindSegmentator.h"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/logger/BufferedConfigurableLogger.h"
#include "cir/common/logger/NullLogger.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <string>
#include <list>
#include "ConfigHelper.h"

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::cpuprocessing;
using namespace cir::gpuprocessing;

void showRecognitionResults(RegistrationPlateRecognizor& recognizor, ImageProcessingService* service);
void recognize(std::string filePath, RegistrationPlateRecognizor& recognizor, ImageProcessingService* service);
void experiment(RegistrationPlateRecognizor& recognizor, ImageProcessingService* service);

int main(int argc, char** argv) {
	ConfigHelper config(argc, argv);

	std::list<std::string> loggerConf;
	BufferedConfigurableLogger logger(loggerConf);
	ImageProcessingService* service = config.getService(logger);

	RegistrationPlateRecognizor recognizor(*service);
	recognizor.setWriteLetters(config.isWriteLetters());

	if(config.keyExists("file")) {
		recognize(config.getPlateFilePath(), recognizor, service);
	} else {
		showRecognitionResults(recognizor, service);
	}
//	experiment(recognizor, &service);

	logger.flushBuffer();
}

void showRecognitionResults(RegistrationPlateRecognizor& recognizor, ImageProcessingService* service) {
	cv::namedWindow("result");
	recognize(getTestFile("registration-plate", "damian.bmp"), recognizor, service);
	recognize(getTestFile("registration-plate", "pt-cruiser-front.jpeg"), recognizor, service);
	recognize(getTestFile("registration-plate", "pt-cruiser-back.jpeg"), recognizor, service);
	recognize(getTestFile("registration-plate", "Audi-Q7-white.jpg"), recognizor, service);
	recognize(getTestFile("registration-plate", "Audi-Q7-black.jpg"), recognizor, service);
	recognize(getTestFile("registration-plate", "skoda.jpeg"), recognizor, service);
	recognize(getTestFile("registration-plate", "passat.jpg"), recognizor, service);
	recognize(getTestFile("registration-plate", "passat2.jpg"), recognizor, service);
	recognize(getTestFile("registration-plate", "audi-a4.jpg"), recognizor, service);
	recognize(getTestFile("registration-plate", "audi-a42.jpg"), recognizor, service);
	recognize(getTestFile("registration-plate", "vectra.jpg"), recognizor, service);
	recognize(getTestFile("registration-plate", "vectra2.jpg"), recognizor, service);
}

void experiment(RegistrationPlateRecognizor& recognizor, ImageProcessingService* service) {
	int reps = 100;
	for(int i = 0; i < reps; i++) {
		recognize(getTestFile("registration-plate", "damian.bmp"), recognizor, service);
	}
	for(int i = 0; i < reps; i++) {
		recognize(getTestFile("registration-plate", "skoda.jpeg"), recognizor, service);
	}
	for(int i = 0; i < reps; i++) {
		recognize(getTestFile("registration-plate", "audi-a4.jpg"), recognizor, service);
	}
}

void recognize(std::string filePath, RegistrationPlateRecognizor& recognizor, ImageProcessingService* service) {
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
