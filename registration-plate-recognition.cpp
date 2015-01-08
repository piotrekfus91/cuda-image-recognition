#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cir::common;
using namespace cir::common::logger;
using namespace cir::common::recognition;
using namespace cir::cpuprocessing;

int main() {
	ImmediateConsoleLogger logger;
	CpuImageProcessingService service(logger);

	RegistrationPlateRecognizor recognizor(service);

	cv::Mat damianMat = cv::imread(getTestFile("registration-plate", "damian.bmp"));
	MatWrapper damianMw(damianMat);
	recognizor.recognize(damianMw);

	cv::Mat ptCruiserFrontMat = cv::imread(getTestFile("registration-plate", "pt-cruiser-front.jpeg"));
	MatWrapper ptCruiserFrontMw(ptCruiserFrontMat);
	recognizor.recognize(ptCruiserFrontMw);

	cv::Mat ptCruiserBackMat = cv::imread(getTestFile("registration-plate", "pt-cruiser-back.jpeg"));
	MatWrapper ptCruiserBackMw(ptCruiserBackMat);
	recognizor.recognize(ptCruiserBackMw);
}
