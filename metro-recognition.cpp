#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/recognition/MetroRecognizor.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::cpuprocessing;

void recognize(std::string filePath, MetroRecognizor& recognizor, ImageProcessingService* service);

int main() {
	ImmediateConsoleLogger logger;
	CpuImageProcessingService service(logger);

	MetroRecognizor recognizor(service);
	recognizor.learn(getTestFile("metro", "metro.png").c_str());

	cv::namedWindow("result");
	recognize(getTestFile("metro", "metro.png"), recognizor, &service);
	recognize(getTestFile("metro", "warszawa_metro_swietokrzyska.jpeg"), recognizor, &service);
	recognize(getTestFile("metro", "metro_warszawa_450.jpeg"), recognizor, &service);
}

void recognize(std::string filePath, MetroRecognizor& recognizor, ImageProcessingService* service) {
	cv::Mat mat = cv::imread(filePath.c_str());
	MatWrapper mw(mat);
	RecognitionInfo recognitionInfo = recognizor.recognize(mw);
	if(recognitionInfo.isSuccess()) {
		mw = service->mark(mw, recognitionInfo.getMatchedSegments());
	}
	cv::imshow("result", mw.getMat());
	cv::waitKey(0);
}
