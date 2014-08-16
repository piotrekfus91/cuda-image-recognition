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
//	cv::namedWindow("Test GPU", CV_WINDOW_AUTOSIZE);

	while(true) {
		capture >> frame;
		gpuFrame.upload(frame);

		cir::common::MatWrapper matWrapper(frame);
		matWrapper = service.bgrToHsv(matWrapper);
		matWrapper = service.detectColorHsv(matWrapper,
				45, 75,
				0.3, 1,
				0.3, 1);
		matWrapper = service.hsvToBgr(matWrapper);

//		cir::common::MatWrapper gpuMatWrapper(gpuFrame);
//		gpuMatWrapper = gpuService.bgrToHsv(gpuMatWrapper);
//		gpuMatWrapper = gpuService.detectColorHsv(gpuMatWrapper,
//				0, 30,
//				0, 1,
//				0, 1);

		imshow("Orig", frame);
		imshow("Test CPU", matWrapper.getMat());
//		imshow("Test GPU", cv::Mat(gpuMatWrapper.getGpuMat()));

		char c = (char)cv::waitKey(30);
		if (c == 27) break;
	}

    return EXIT_SUCCESS;
}
