#include <iostream>
#include <cstdlib>

#include "opencv2/opencv.hpp"
#include "cir/cpuprocessing/CpuImageProcessingService.h"

using namespace std;

int main(int argc, char** argv) {
	cir::cpuprocessing::CpuImageProcessingService service;
	cv::VideoCapture capture(0);
	cv::Mat frame;

	cv::namedWindow("Test", CV_WINDOW_AUTOSIZE);

	while(true) {
		capture >> frame;

		cir::common::MatWrapper matWrapper(frame);
		matWrapper = service.toGrey(matWrapper);

		imshow("Test", matWrapper.getMat());

		char c = (char)cv::waitKey(30);
		if (c == 27) break;
	}

    return EXIT_SUCCESS;
}
