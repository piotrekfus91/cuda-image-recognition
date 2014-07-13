#include <iostream>
#include <cstdlib>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"

using namespace std;

int main(int argc, char** argv) {
	cout << cv::gpu::getCudaEnabledDeviceCount() << endl;
	cir::cpuprocessing::CpuImageProcessingService service;
	cir::gpuprocessing::GpuImageProcessingService gpuService;
	cv::VideoCapture capture(0);
	cv::Mat frame;

	cv::gpu::GpuMat gpuFrame;

	cv::namedWindow("Test CPU", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Test GPU", CV_WINDOW_AUTOSIZE);

	while(true) {
		capture >> frame;
		gpuFrame.upload(frame);

		cir::common::MatWrapper matWrapper(frame);
		matWrapper = service.highPass(matWrapper);

		cir::common::MatWrapper gpuMatWrapper(gpuFrame);
		gpuMatWrapper = gpuService.toGrey(gpuMatWrapper);
		gpuMatWrapper = gpuService.highPass(gpuMatWrapper);

		imshow("Test CPU", matWrapper.getMat());
		imshow("Test GPU", cv::Mat(gpuMatWrapper.getGpuMat()));

		char c = (char)cv::waitKey(30);
		if (c == 27) break;
	}

    return EXIT_SUCCESS;
}
