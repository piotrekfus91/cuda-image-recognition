#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/cpuprocessing/CpuUnionFindSegmentator.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuUnionFindSegmentator.h"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/logger/NullLogger.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "cir/common/recognition/RegistrationPlateTeacher.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::cpuprocessing;
using namespace cir::gpuprocessing;

void recognize(std::string filePath, RegistrationPlateRecognizor& recognizor, ImageProcessingService* service);

int main() {
	cv::Mat mat = cv::imread(getTestFile("cpu-processing", "metro_red_yellow.bmp").c_str());

	NullLogger logger;
	CpuImageProcessingService cpuService(logger);
	cpuService.setSegmentator(new CpuUnionFindSegmentator);

	cpuService.init(mat.cols, mat.rows);

	RegistrationPlateRecognizor recognizor(cpuService);
	RegistrationPlateTeacher teacher(&recognizor);
	recognizor.setWriteLetters(true);
	teacher.teach(getTestFile("registration-plate", "alphabet"));

	cv::namedWindow("result");
	recognize(getTestFile("registration-plate", "damian.bmp"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "pt-cruiser-front.jpeg"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "pt-cruiser-back.jpeg"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "Audi-Q7-white.jpg"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "Audi-Q7-black.jpg"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "skoda.jpeg"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "passat.jpg"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "passat2.jpg"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "audi-a4.jpg"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "audi-a42.jpg"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "vectra.jpg"), recognizor, &cpuService);
	recognize(getTestFile("registration-plate", "vectra2.jpg"), recognizor, &cpuService);
}

void recognize(std::string filePath, RegistrationPlateRecognizor& recognizor, ImageProcessingService* service) {
	cv::Mat mat = cv::imread(filePath.c_str());
	MatWrapper mw(mat);
	service->init(mw.getWidth(), mw.getHeight());
	RecognitionInfo recognitionInfo = recognizor.recognize(mw);

	if(recognitionInfo.isSuccess()) {
		mw = service->mark(mw, recognitionInfo.getMatchedSegments());
	}
	cv::imshow("result", mw.getMat());
	cv::waitKey(0);
}
