#include "cir/common/ColorDetector.h"

namespace cir { namespace common {

ColorDetector::ColorDetector() {

}

ColorDetector::~ColorDetector() {

}

MatWrapper ColorDetector::detectColorHsv(const MatWrapper& input, const int hueNumber,
		const double* minHues, const double* maxHues, const double minSaturation,
		const double maxSaturation,	const double minValue, const double maxValue) {
	int* iMinHues = (int*) malloc(sizeof(int) * hueNumber);
	int* iMaxHues = (int*) malloc(sizeof(int) * hueNumber);
	for(int i = 0; i < hueNumber; i++) {
		iMinHues[i] = minHues[i] / 2;
		iMaxHues[i] = maxHues[i] / 2;
	}

	const int iMinSaturation = 255 * minSaturation;
	const int iMaxSaturation = 255 * maxSaturation;
	const int iMinValue = 255 * minValue;
	const int iMaxValue = 255 * maxValue;

	MatWrapper inputMatWrapper = input;

	return doDetectColor(inputMatWrapper, hueNumber, iMinHues, iMaxHues, iMinSaturation, iMaxSaturation,
			iMinValue, iMaxValue);
}

}}
