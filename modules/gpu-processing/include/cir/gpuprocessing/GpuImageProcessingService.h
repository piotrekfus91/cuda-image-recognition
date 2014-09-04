#ifndef GPUIMAGEPROCESSINGSERVICE_H_
#define GPUIMAGEPROCESSINGSERVICE_H_

#include "cir/common/ImageProcessingService.h"
#include "cir/gpuprocessing/GpuColorDetector.h"
#include "cir/gpuprocessing/GpuRegionSplittingSegmentator.h"

namespace cir { namespace gpuprocessing {

class GpuImageProcessingService : public cir::common::ImageProcessingService {
public:
	GpuImageProcessingService();
	virtual ~GpuImageProcessingService();

	virtual cir::common::MatWrapper toGrey(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper threshold(const cir::common::MatWrapper& input, double thresholdValue);
	virtual cir::common::MatWrapper lowPass(const cir::common::MatWrapper& input, int size = DEFAULT_LOW_PASS_KERNEL_SIZE);
	virtual cir::common::MatWrapper highPass(const cir::common::MatWrapper& input, int size = 1);
	virtual cir::common::MatWrapper bgrToHsv(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper hsvToBgr(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper detectColorHsv(const cir::common::MatWrapper& input, const int hueNumber,
			const double* minHues, const double* maxHues, const double minSaturation,
			const double maxSaturation,	const double minValue, const double maxValue);
	virtual cir::common::SegmentArray* segmentate(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper mark(cir::common::MatWrapper& input, cir::common::SegmentArray* segmentArray);
	virtual cir::common::MatWrapper crop(cir::common::MatWrapper& input, cir::common::Segment* segment);

private:
	GpuColorDetector _gpuColorDetector;
	GpuRegionSplittingSegmentator _segmentator;
};

}}
#endif
