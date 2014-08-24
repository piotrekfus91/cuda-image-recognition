#ifndef CPUCOLORDETECTOR_H_
#define CPUCOLORDETECTOR_H_

#include "cir/common/ColorDetector.h"

namespace cir { namespace cpuprocessing {

class CpuColorDetector : public cir::common::ColorDetector {
public:
	CpuColorDetector();
	virtual ~CpuColorDetector();

protected:
	virtual cir::common::MatWrapper doDetectColor(cir::common::MatWrapper& input, const int hueNumber,
			const int* minHues,	const int* maxHues, const int minSat, const int maxSat,
			const int minValue, const int maxValue);
};

}}
#endif
