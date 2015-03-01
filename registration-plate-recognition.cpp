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
	cv::gpu::GpuMat gpuMat(mat);

	NullLogger logger;
	GpuImageProcessingService gpuService(logger);
	gpuService.setSegmentator(new GpuUnionFindSegmentator);

	gpuService.init(gpuMat.cols, gpuMat.rows);

	RegistrationPlateRecognizor recognizor(gpuService);
	RegistrationPlateTeacher teacher(&recognizor);
	teacher.teach(getTestFile("registration-plate", "alphabet"));

	cv::namedWindow("result");
	recognize(getTestFile("registration-plate", "damian.bmp"), recognizor, &gpuService);
	recognize(getTestFile("registration-plate", "pt-cruiser-front.jpeg"), recognizor, &gpuService);
	recognize(getTestFile("registration-plate", "pt-cruiser-back.jpeg"), recognizor, &gpuService);
	recognize(getTestFile("registration-plate", "Audi-Q7-white.jpg"), recognizor, &gpuService);
	recognize(getTestFile("registration-plate", "Audi-Q7-black.jpg"), recognizor, &gpuService);
}

void recognize(std::string filePath, RegistrationPlateRecognizor& recognizor, ImageProcessingService* service) {
	cv::Mat mat = cv::imread(filePath.c_str());
	cv::gpu::GpuMat gpuMat(mat);
	MatWrapper mw(gpuMat);
	service->init(mw.getWidth(), mw.getHeight());
	RecognitionInfo recognitionInfo = recognizor.recognize(mw);

	if(recognitionInfo.isSuccess()) {
		mw = service->mark(mw, recognitionInfo.getMatchedSegments());
		cv::Mat mat;
		cv::gpu::GpuMat gpuMat = mw.getGpuMat();
		gpuMat.download(mat);
		cv::imshow("result", mat);
		cv::waitKey(0);
	}
}
