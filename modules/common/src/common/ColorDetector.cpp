#include "cir/common/ColorDetector.h"

namespace cir { namespace common {

ColorDetector::ColorDetector() {

}

ColorDetector::~ColorDetector() {

}

MatWrapper ColorDetector::detectColorHsv(const MatWrapper& input, const double minHue,
				const double maxHue, const double minSaturation, const double maxSaturation,
				const double minValue, const double maxValue) {
	const int iMinHue = minHue / 2;
	const int iMaxHue = maxHue / 2;
	const int iMinSaturation = 255 * minSaturation;
	const int iMaxSaturation = 255 * maxSaturation;
	const int iMinValue = 255 * minValue;
	const int iMaxValue = 255 * maxValue;

	MatWrapper inputMatWrapper = input;

	return doDetectColor(inputMatWrapper, iMinHue, iMaxHue, iMinSaturation, iMaxSaturation,
			iMinValue, iMaxValue);
}

}}
