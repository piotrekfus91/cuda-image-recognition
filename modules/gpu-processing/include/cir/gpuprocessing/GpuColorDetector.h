#ifndef GPUCOLORDETECTOR_H_
#define GPUCOLORDETECTOR_H_

#include "cir/common/ColorDetector.h"

namespace cir { namespace gpuprocessing {

class GpuColorDetector : public cir::common::ColorDetector {
public:
	GpuColorDetector();
	virtual ~GpuColorDetector();

protected:
	virtual cir::common::MatWrapper doDetectColor(cir::common::MatWrapper& input, const int minHue,
			const int maxHue, const int minSat, const int maxSat, const int minValue,
			const int maxValue);
};

}}
#endif
