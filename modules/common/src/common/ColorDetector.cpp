#include "cir/common/ColorDetector.h"

namespace cir { namespace common {

ColorDetector::ColorDetector() {

}

ColorDetector::~ColorDetector() {

}

MatWrapper ColorDetector::detectColorHsv(const MatWrapper& input, const int hsvRangesNumber, const HsvRange* hsvRanges) {
	OpenCvHsvRange* openCvHsvRanges = (OpenCvHsvRange*) malloc(sizeof(OpenCvHsvRange) * hsvRangesNumber);
	for(int i = 0; i < hsvRangesNumber; i++) {
		openCvHsvRanges[i].less.hue = hsvRanges[i].less.hue / 2;
		openCvHsvRanges[i].less.saturation = hsvRanges[i].less.saturation * 255;
		openCvHsvRanges[i].less.value = hsvRanges[i].less.value * 255;

		openCvHsvRanges[i].greater.hue = hsvRanges[i].greater.hue / 2;
		openCvHsvRanges[i].greater.saturation = hsvRanges[i].greater.saturation * 255;
		openCvHsvRanges[i].greater.value = hsvRanges[i].greater.value * 255;
	}

	MatWrapper inputMatWrapper = input;

	return doDetectColor(inputMatWrapper, hsvRangesNumber, openCvHsvRanges);
}

}}
