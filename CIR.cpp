#include <cstdlib>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/common/cuda_host_util.cuh"

using namespace std;

void imgCpu(const char*);
void imgGpu(const char*);
void cam();

int main(int argc, char** argv) {
	imgCpu("sample.bmp");
	imgGpu("sample.bmp");
//	cam();

    return EXIT_SUCCESS;
}

void imgCpu(const char* fileName) {
	cir::cpuprocessing::CpuImageProcessingService cpuService;
	cv::Mat img = cv::imread(fileName, CV_LOAD_IMAGE_UNCHANGED);
	cv::imshow("ORIG", img);

	cir::common::MatWrapper mw(img);

	cpuService.init(img.cols, img.rows);

	mw = cpuService.toGrey(mw);

	double* hu = cpuService.countHuMoments(mw);
	for(int i = 0; i < 7; i++) {
		cout << i << ": " << hu[i] << endl;
	}

	cv::namedWindow("ORIG");
	cv::namedWindow("CPU");

	cv::imshow("CPU", cv::Mat(mw.getMat()));
	cv::waitKey(0);
}

void imgGpu(const char* fileName) {
	cir::gpuprocessing::GpuImageProcessingService gpuService;
	cv::Mat img = cv::imread(fileName, CV_LOAD_IMAGE_UNCHANGED);
	cv::imshow("ORIG", img);

	cv::gpu::GpuMat gpuMat(img);
	cir::common::MatWrapper mw(gpuMat);

	gpuService.init(img.cols, img.rows);

	mw = gpuService.toGrey(mw);
	double* hu = gpuService.countHuMoments(mw);
	for(int i = 0; i < 7; i++) {
		cout << i << ": " << hu[i] << endl;
	}

	cv::namedWindow("ORIG");
	cv::namedWindow("GPU");

	cv::imshow("GPU", cv::Mat(mw.getGpuMat()));
	cv::waitKey(0);
}

void cam() {
	cir::common::cuda_init();

	cir::cpuprocessing::CpuImageProcessingService service;
	cir::gpuprocessing::GpuImageProcessingService gpuService;
	cv::VideoCapture capture(0);
	cv::Mat frame;

	cv::gpu::GpuMat gpuFrame;

	cv::namedWindow("Test CPU", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Test GPU", CV_WINDOW_AUTOSIZE);

	capture >> frame;
	service.init(frame.cols, frame.rows);
	gpuService.init(frame.cols, frame.rows);

	int i = 0;
	while(true) {
		capture >> frame;
		gpuFrame.upload(frame);

		double minHues[2] = {45, 345};
		double maxHues[2] = {75, 15};

		cir::common::MatWrapper matWrapper(frame);
		matWrapper = service.bgrToHsv(matWrapper);
		matWrapper = service.detectColorHsv(matWrapper, 2,
				minHues, maxHues,
				0, 1,
				0, 1);
		service.segmentate(matWrapper);
//		matWrapper = service.mark(matWrapper, segmentArray);
		matWrapper = service.hsvToBgr(matWrapper);
//		matWrapper = service.crop(matWrapper, segmentArray->segments[0]);

		cir::common::MatWrapper gpuMatWrapper(gpuFrame);
//		gpuMatWrapper = gpuService.bgrToHsv(gpuMatWrapper);
//		gpuMatWrapper = gpuService.detectColorHsv(gpuMatWrapper, 2,
//				minHues, maxHues,
//				0, 1,
//				0, 1);
//		gpuService.segmentate(gpuMatWrapper);
//		gpuMatWrapper = gpuService.hsvToBgr(gpuMatWrapper);

		imshow("Orig", frame);
		imshow("Test CPU", matWrapper.getMat());
		imshow("Test GPU", cv::Mat(gpuMatWrapper.getGpuMat()));

		char c = (char)cv::waitKey(30);
		if (c == 27) break;
	}

	cir::common::cuda_shutdown();
}
