#include "cir/cpuprocessing/CpuColorDetector.h"
#include <vector>

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuColorDetector::CpuColorDetector() {

}

CpuColorDetector::~CpuColorDetector() {

}

MatWrapper CpuColorDetector::doDetectColor(const MatWrapper& input, const int hsvRangesNumber,
		const OpenCvHsvRange* hsvRanges) {
	cv::Mat outputMat = input.getMat().clone();

	int cols = outputMat.cols;
	int rows = outputMat.rows;

	bool clear;

	for(int x = 0; x < cols; x++) {
		for(int y = 0; y < rows; y++) {
			cv::Vec3b& hsv = outputMat.at<cv::Vec3b>(y, x);
			clear = true;

			int hue = hsv[0];
			int saturation = hsv[1];
			int value = hsv[2];

			for(int i = 0; i < hsvRangesNumber; i++) {
				OpenCvHsvRange hsvRange = hsvRanges[i];
				OpenCvHsv less = hsvRange.less;
				OpenCvHsv greater = hsvRange.greater;

				if(saturation >= less.saturation && saturation <= greater.saturation
						&& value >= less.value && value <= greater.value) {
					if(less.hue <= greater.hue) {
						if(hue >= less.hue && hue <= greater.hue) {
							clear = false;
							break;
						}
					} else {
						if(hue >= less.hue || hue <= greater.hue) {
							clear = false;
							break;
						}
					}

				}
			}

			if(clear) {
				hsv[0] = 0;
				hsv[1] = 0;
				hsv[2] = 0;
			}
		}
	}

	MatWrapper outputMw(outputMat);
	outputMw.setColorScheme(input.getColorScheme());
	return outputMw;
}

}}
