#include "cir/cpuprocessing/CpuColorDetector.h"
#include <vector>

using namespace cir::common;

namespace cir { namespace cpuprocessing {

CpuColorDetector::CpuColorDetector() {

}

CpuColorDetector::~CpuColorDetector() {

}

MatWrapper CpuColorDetector::doDetectColor(MatWrapper& input, const int hueNumber,
		const int* minHues, const int* maxHues,	const int minSat, const int maxSat,
		const int minValue, const int maxValue) {
	cv::Mat output(input.getMat());

	int cols = output.cols;
	int rows = output.rows;

	bool clear;

	for(int x = 0; x < cols; x++) {
		for(int y = 0; y < rows; y++) {
			cv::Vec3b& hsv = output.at<cv::Vec3b>(y, x);
			clear = true;

			if(hsv[1] >= minSat && hsv[1] <= maxSat
					&& hsv[2] >= minValue && hsv[2] <= maxValue) {

				for(int hueIndex = 0; hueIndex < hueNumber; hueIndex++) {
					int hue = hsv[0];
					int minHue = minHues[hueIndex];
					int maxHue = maxHues[hueIndex];

					if(minHue <= maxHue) {
						if(hue >= minHue && hue <= maxHue) {
							clear = false;
							break;
						}
					} else {
						if(hue >= minHue || hue <= maxHue) {
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

	return output;
}

}}
