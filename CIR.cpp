#include <cstdlib>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/common/cuda_host_util.cuh"
#include "cir/common/logger/ImmediateConsoleLogger.h"
#include "cir/common/test_file_loader.h"

using namespace std;

void imgCpu(const char*, cir::common::logger::Logger&);
void imgGpu(const char*, cir::common::logger::Logger&);
void cam(cir::common::logger::Logger&);

int main(int argc, char** argv) {
	cir::common::logger::ImmediateConsoleLogger logger;
	cir::cpuprocessing::CpuImageProcessingService cpuService(logger);

	cv::Mat dMat = cv::imread(cir::common::getTestFile("registration-plate/alphabet", "D.bmp").c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	cir::common::MatWrapper dMw(dMat);
	dMw = cpuService.threshold(dMw, true, 127);
	double* dHuMoments = cpuService.countHuMoments(dMw);

	cv::Mat ddMat = cv::imread(cir::common::getTestFile("registration-plate/alphabet", "DD.bmp").c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	cir::common::MatWrapper ddMw(ddMat);
	ddMw = cpuService.threshold(ddMw, true, 127);
	double* ddHuMoments = cpuService.countHuMoments(ddMw);

	cv::namedWindow("d");
	cv::imshow("d", dMw.getMat());

	cv::namedWindow("dd");
	cv::imshow("dd", ddMw.getMat());

	cv::Moments dMoments = cv::moments(dMat, true);
	cv::Moments ddMoments = cv::moments(ddMat, true);

	std::cout << dMoments.m00 << std::endl;
	std::cout << dMoments.m01 << std::endl;
	std::cout << dMoments.m10 << std::endl;
	std::cout << dMoments.m11 << std::endl;
	std::cout << dMoments.m02 << std::endl;
	std::cout << dMoments.m20 << std::endl;
	std::cout << dMoments.m21 << std::endl;
	std::cout << dMoments.m12 << std::endl;
	std::cout << dMoments.m30 << std::endl;
	std::cout << dMoments.m03 << std::endl;

	double* dOHuMoments = new double[7];
	double* ddOHuMoments = new double[7];

	cv::HuMoments(dMoments, dOHuMoments);
	cv::HuMoments(ddMoments, ddOHuMoments);
	for(int i = 0; i < 7; i++) {
		double ratio = dHuMoments[i] / ddHuMoments[i];
		double oRatio = dOHuMoments[i] / ddOHuMoments[i];
		std::cout << i << std::endl << dHuMoments[i] << std::endl << ddHuMoments[i] << std::endl << ratio << std::endl;
		std::cout << dOHuMoments[i] << std::endl << ddOHuMoments[i] << std::endl << oRatio << std::endl;
		std::cout << std::endl;
	}

	cv::waitKey(0);

//	imgGpu(cir::common::getTestFile("cpu-processing", "dashes.bmp").c_str(), logger);
//	cam();

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
