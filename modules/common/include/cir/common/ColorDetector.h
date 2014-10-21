#ifndef COLORDETECTOR_H_
#define COLORDETECTOR_H_

#include "cir/common/MatWrapper.h"
#include "cir/common/Hsv.h"

namespace cir { namespace common {

class ColorDetector {
public:
	ColorDetector();
	virtual ~ColorDetector();

	virtual MatWrapper detectColorHsv(const cir::common::MatWrapper& input, const int hsvRangesNumber,
			const cir::common::HsvRange* hsvRanges);

protected:
	virtual MatWrapper doDetectColor(const cir::common::MatWrapper& input, const int hsvRangesNumber,
			const cir::common::OpenCvHsvRange* hsvRanges) = 0;
};

}}
#endif
