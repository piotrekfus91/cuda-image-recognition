#include "cir/cpuprocessing/CpuColorDetector.h"
#include <vector>
#include <iostream>

namespace cir { namespace cpuprocessing {

CpuColorDetector::CpuColorDetector() {

}

CpuColorDetector::~CpuColorDetector() {

}

cv::Mat CpuColorDetector::detectColor(cv::Mat& input, const int minHue, const int maxHue,
			const int minSat, const int maxSat, const int minValue, const int maxValue) {
	cv::Mat output(input);

	int cols = input.cols;
	int rows = input.rows;

	for(int x = 0; x < cols; x++) {
		for(int y = 0; y < rows; y++) {
			cv::Vec3b& hsv = output.at<cv::Vec3b>(y, x);

			if(hsv[0] < minHue || hsv[0] > maxHue
					|| hsv[1] < minSat || hsv[1] > maxSat
					|| hsv[2] < minValue || hsv[2] > maxValue) {
				hsv[0] = 0;
				hsv[1] = 0;
				hsv[2] = 0;
			}
		}
	}

	return output;
}

}}
