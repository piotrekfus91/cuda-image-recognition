#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "cir/common/recognition/RegistrationPlateTeacher.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::cpuprocessing;

void recognize(std::string filePath, RegistrationPlateRecognizor& recognizor, ImageProcessingService* service);

int main() {
	ImmediateConsoleLogger logger;
	CpuImageProcessingService service(logger);

	RegistrationPlateRecognizor recognizor(service);
	RegistrationPlateTeacher teacher(&recognizor);
	teacher.teach(getTestFile("registration-plate", "alphabet"));

	cv::namedWindow("result");
	recognize(getTestFile("registration-plate", "damian.bmp"), recognizor, &service);
	recognize(getTestFile("registration-plate", "pt-cruiser-front.jpeg"), recognizor, &service);
	recognize(getTestFile("registration-plate", "pt-cruiser-back.jpeg"), recognizor, &service);
	recognize(getTestFile("registration-plate", "Audi-Q7-white.jpg"), recognizor, &service);
	recognize(getTestFile("registration-plate", "Audi-Q7-black.jpg"), recognizor, &service);
}

void recognize(std::string filePath, RegistrationPlateRecognizor& recognizor, ImageProcessingService* service) {
	cv::Mat mat = cv::imread(filePath.c_str());
	MatWrapper mw(mat);
	RecognitionInfo recognitionInfo = recognizor.recognize(mw);
	if(recognitionInfo.isSuccess()) {
		mw = service->mark(mw, recognitionInfo.getMatchedSegments());
	}
	cv::imshow("result", mw.getMat());
	cv::waitKey(0);
}
