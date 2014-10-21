#ifndef CPUCOLORDETECTOR_H_
#define CPUCOLORDETECTOR_H_

#include "cir/common/ColorDetector.h"

namespace cir { namespace cpuprocessing {

class CpuColorDetector : public cir::common::ColorDetector {
public:
	CpuColorDetector();
	virtual ~CpuColorDetector();

protected:
	virtual cir::common::MatWrapper doDetectColor(const cir::common::MatWrapper& input, const int hsvRangesNumber,
			const cir::common::OpenCvHsvRange* hsvRanges);
};

}}
#endif
