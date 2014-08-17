#ifndef COLORDETECTOR_H_
#define COLORDETECTOR_H_

#include "cir/common/MatWrapper.h"

namespace cir { namespace common {

class ColorDetector {
public:
	ColorDetector();
	virtual ~ColorDetector();

	virtual MatWrapper detectColorHsv(const MatWrapper& input, const double minHue,
				const double maxHue, const double minSaturation, const double maxSaturation,
				const double minValue, const double maxValue);

protected:
	virtual MatWrapper doDetectColor(MatWrapper& input, const int minHue, const int maxHue,
				const int minSat, const int maxSat, const int minValue, const int maxValue) = 0;
};

}}
#endif
