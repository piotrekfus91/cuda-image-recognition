#ifndef GPUCOLORDETECTOR_H_
#define GPUCOLORDETECTOR_H_

#include "cir/common/ColorDetector.h"

namespace cir { namespace gpuprocessing {

class GpuColorDetector : public cir::common::ColorDetector {
public:
	GpuColorDetector();
	virtual ~GpuColorDetector();

protected:
	virtual cir::common::MatWrapper doDetectColor(const cir::common::MatWrapper& input, const int hsvRangesNumber,
			const cir::common::OpenCvHsvRange* hsvRanges);
};

}}
#endif
