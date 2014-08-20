#ifndef CPUIMAGEPROCESSINGSERVICE_H_
#define CPUIMAGEPROCESSINGSERVICE_H_

#include "cir/common/ImageProcessingService.h"
#include "cir/cpuprocessing/CpuColorDetector.h"
#include "cir/cpuprocessing/CpuRegionGrowingSegmentator.h"
#include "cir/cpuprocessing/CpuRedMarker.h"

namespace cir { namespace cpuprocessing {

class CpuImageProcessingService : public cir::common::ImageProcessingService {
public:
	CpuImageProcessingService();
	virtual ~CpuImageProcessingService();
	virtual cir::common::MatWrapper toGrey(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper threshold(const cir::common::MatWrapper& input, double thresholdValue);
	virtual cir::common::MatWrapper lowPass(const cir::common::MatWrapper& input, int size = DEFAULT_LOW_PASS_KERNEL_SIZE);
	virtual cir::common::MatWrapper highPass(const cir::common::MatWrapper& input, int size = 1);
	virtual cir::common::MatWrapper bgrToHsv(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper hsvToBgr(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper detectColorHsv(const cir::common::MatWrapper& input, const double minHue,
				const double maxHue, const double minSaturation, const double maxSaturation,
				const double minValue, const double maxValue);
	virtual cir::common::SegmentArray* segmentate(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper mark(cir::common::MatWrapper& input, cir::common::SegmentArray* segmentArray);

private:
	CpuColorDetector _cpuColorDetector;
	CpuRegionGrowingSegmentator _segmentator;
	CpuRedMarker _marker;
};

}}
#endif
