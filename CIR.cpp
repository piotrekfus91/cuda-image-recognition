#include <iostream>
#include <cstdlib>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "cir/cpuprocessing/CpuImageProcessingService.h"
#include "cir/gpuprocessing/GpuImageProcessingService.h"
#include "cir/common/cuda_host_util.cuh"

using namespace std;

int main(int argc, char** argv) {
	cir::common::cuda_init();

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
		matWrapper = service.bgrToHsv(matWrapper);
		matWrapper = service.detectColorHsv(matWrapper,
				45, 75,
				0, 1,
				0, 1);
		cir::common::SegmentArray* segmentArray = service.segmentate(matWrapper);
		matWrapper = service.mark(matWrapper, segmentArray);
		matWrapper = service.hsvToBgr(matWrapper);

		cir::common::MatWrapper gpuMatWrapper(gpuFrame);
		gpuMatWrapper = gpuService.bgrToHsv(gpuMatWrapper);
		gpuMatWrapper = gpuService.detectColorHsv(gpuMatWrapper,
				30, 90,
				0, 1,
				0, 1);
		gpuMatWrapper = gpuService.hsvToBgr(gpuMatWrapper);

		imshow("Orig", frame);
		imshow("Test CPU", matWrapper.getMat());
		imshow("Test GPU", cv::Mat(gpuMatWrapper.getGpuMat()));

		char c = (char)cv::waitKey(30);
		if (c == 27) break;
	}

	cir::common::cuda_shutdown();

    return EXIT_SUCCESS;
}
