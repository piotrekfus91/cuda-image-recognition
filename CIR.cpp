#include <cstdlib>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/cpuprocessing/CpuUnionFindSegmentator.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuUnionFindSegmentator.h"
#include "cir/common/cuda_host_util.cuh"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/logger/NullLogger.h"
#include "cir/common/test_file_loader.h"
#include "cir/common/video/SingleThreadVideoHandler.h"
#include "cir/common/video/MultiThreadVideoHandler.h"
#include "cir/common/video/RecognitionVideoConverter.h"
#include "cir/common/recognition/RegistrationPlateRecognizor.h"
#include "cir/common/recognition/RegistrationPlateTeacher.h"
#include "cir/devenv/ThreadInfo.h"

using namespace std;

void imgCpu(const char*, cir::common::logger::Logger&);
void imgGpu(const char*, cir::common::logger::Logger&);
void cam(cir::common::logger::Logger&);

int main(int argc, char** argv) {
	cir::common::logger::NullLogger logger;
//	cir::gpuprocessing::GpuImageProcessingService cpuService(logger);
//	cpuService.setSegmentator(new cir::gpuprocessing::GpuUnionFindSegmentator);
	cir::cpuprocessing::CpuImageProcessingService cpuService(logger);
	cpuService.setSegmentator(new cir::cpuprocessing::CpuUnionFindSegmentator);

	cir::common::recognition::RegistrationPlateRecognizor* recognizor
			= new cir::common::recognition::RegistrationPlateRecognizor(cpuService);
	cir::common::recognition::RegistrationPlateTeacher teacher(recognizor);
	teacher.teach(cir::common::getTestFile("registration-plate", "alphabet"));

//	cir::common::video::VideoHandler* videoHandler = new cir::common::video::SingleThreadVideoHandler();
	cir::common::video::VideoHandler* videoHandler = new cir::common::video::MultiThreadVideoHandler();
	cir::common::video::RecognitionVideoConverter* videoConverter
			= new cir::common::video::RecognitionVideoConverter(recognizor, &cpuService);
	std::string inputFilePath = cir::common::getTestFile("video", "walk.avi");
	videoHandler->handle(inputFilePath, videoConverter);

    return EXIT_SUCCESS;
}

void imgCpu(const char* fileName, cir::common::logger::Logger& logger) {
	cir::cpuprocessing::CpuImageProcessingService cpuService(logger);
	cv::Mat img = cv::imread(fileName, CV_LOAD_IMAGE_UNCHANGED);
	cv::imshow("ORIG", img);

	cir::common::MatWrapper mw(img);

	cpuService.init(img.cols, img.rows);

	cir::common::Hsv lessRed;
	lessRed.hue = 345;
	lessRed.saturation = 0.2;
	lessRed.value = 0.2;
	cir::common::Hsv greaterRed;
	greaterRed.hue = 15;
	greaterRed.saturation = 1;
	greaterRed.value = 1;
	cir::common::HsvRange rangeRed;
	rangeRed.less = lessRed;
	rangeRed.greater = greaterRed;

	cir::common::Hsv lessYellow;
	lessYellow.hue = 45;
	lessYellow.saturation = 0.2;
	lessYellow.value = 0.2;
	cir::common::Hsv greaterYellow;
	greaterYellow.hue = 75;
	greaterYellow.saturation = 1;
	greaterYellow.value = 1;
	cir::common::HsvRange rangeYellow;
	rangeYellow.less = lessYellow;
	rangeYellow.greater = greaterYellow;

	cir::common::HsvRange hsvRanges[2] = {rangeRed, rangeYellow};

	mw = cpuService.bgrToHsv(mw);
	mw = cpuService.detectColorHsv(mw, 2, hsvRanges);
	cir::common::SegmentArray* segmentArray = cpuService.segmentate(mw);
	mw = cpuService.hsvToBgr(mw);
	std::cerr << "total: " << segmentArray->size << std::endl;
	cv::imwrite("metro_red_yellow.bmp", mw.getMat());
	mw = cpuService.mark(mw, segmentArray);

	cv::namedWindow("ORIG");
	cv::namedWindow("CPU");

	cv::imshow("CPU", cv::Mat(mw.getMat()));
	cv::waitKey(0);
}

void imgGpu(const char* fileName, cir::common::logger::Logger& logger) {
	cir::gpuprocessing::GpuImageProcessingService gpuService(logger);
	cv::Mat img = cv::imread(fileName, CV_LOAD_IMAGE_UNCHANGED);
	cv::gpu::GpuMat gpuImg(img);
	cv::imshow("ORIG", img);

	cir::common::MatWrapper mw(gpuImg);

	gpuService.init(img.cols, img.rows);

	cir::common::Hsv lessRed;
	lessRed.hue = 345;
	lessRed.saturation = 0.2;
	lessRed.value = 0.2;
	cir::common::Hsv greaterRed;
	greaterRed.hue = 15;
	greaterRed.saturation = 1;
	greaterRed.value = 1;
	cir::common::HsvRange rangeRed;
	rangeRed.less = lessRed;
	rangeRed.greater = greaterRed;

	cir::common::Hsv lessYellow;
	lessYellow.hue = 45;
	lessYellow.saturation = 0.2;
	lessYellow.value = 0.2;
	cir::common::Hsv greaterYellow;
	greaterYellow.hue = 75;
	greaterYellow.saturation = 1;
	greaterYellow.value = 1;
	cir::common::HsvRange rangeYellow;
	rangeYellow.less = lessYellow;
	rangeYellow.greater = greaterYellow;

	cir::common::HsvRange hsvRanges[2] = {rangeRed, rangeYellow};

	mw = gpuService.bgrToHsv(mw);
	mw = gpuService.detectColorHsv(mw, 2, hsvRanges);
	cir::common::SegmentArray* segmentArray = gpuService.segmentate(mw);
	mw = gpuService.hsvToBgr(mw);
//	cv::imwrite("metro_red_yellow.bmp", mw.getMat());

	std::cerr << "totasdfl: " << segmentArray->size << std::endl;

	gpuImg.download(img);
	cir::common::MatWrapper mw2(img);
	cir::cpuprocessing::CpuImageProcessingService cpuService(logger);
	mw2 = cpuService.mark(mw2, segmentArray);

	cv::namedWindow("ORIG");
	cv::namedWindow("CPU");

	cv::namedWindow("grey");
	cv::namedWindow("median");

	mw = gpuService.toGrey(mw);
	gpuImg = mw.getGpuMat();
	gpuImg.download(img);
	cv::imshow("grey", img);

	mw = gpuService.median(mw);
	gpuImg = mw.getGpuMat();
	gpuImg.download(img);
	cv::imshow("median", img);


	cv::imshow("CPU", cv::Mat(mw2.getMat()));
	cv::waitKey(0);
}

void cam(cir::common::logger::Logger& logger) {
	cir::common::cuda_init();

	cir::cpuprocessing::CpuImageProcessingService service(logger);
	cir::gpuprocessing::GpuImageProcessingService gpuService(logger);
	cv::VideoCapture capture(0);
	cv::Mat frame;

	cv::gpu::GpuMat gpuFrame;

	cv::namedWindow("Test CPU", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Test GPU", CV_WINDOW_AUTOSIZE);

	capture >> frame;
	service.init(frame.cols, frame.rows);
	gpuService.init(frame.cols, frame.rows);

	while(true) {
		capture >> frame;
		gpuFrame.upload(frame);



		imshow("Orig", frame);
//		imshow("Test CPU", matWrapper.getMat());
//		imshow("Test GPU", cv::Mat(gpuMatWrapper.getGpuMat()));

		char c = (char)cv::waitKey(30);
		if (c == 27) break;
	}

	cir::common::cuda_shutdown();
}
