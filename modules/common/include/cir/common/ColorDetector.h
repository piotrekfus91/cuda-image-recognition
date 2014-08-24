#ifndef COLORDETECTOR_H_
#define COLORDETECTOR_H_

#include "cir/common/MatWrapper.h"

namespace cir { namespace common {

class ColorDetector {
public:
	ColorDetector();
	virtual ~ColorDetector();

	virtual MatWrapper detectColorHsv(const MatWrapper& input, const int hueNumber,
			const double* minHues, const double* maxHues, const double minSaturation,
			const double maxSaturation,	const double minValue, const double maxValue);

protected:
	virtual MatWrapper doDetectColor(MatWrapper& input, const int hueNumber, const int* minHues,
			const int* maxHues,	const int minSat, const int maxSat, const int minValue,
			const int maxValue) = 0;
};

}}
#endif
