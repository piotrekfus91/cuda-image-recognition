#ifndef GPUIMAGEPROCESSINGSERVICE_H_
#define GPUIMAGEPROCESSINGSERVICE_H_

#include "cir/common/ImageProcessingService.h"
#include "cir/gpuprocessing/GpuColorDetector.h"
#include "cir/gpuprocessing/GpuRegionSplittingSegmentator.h"
#include "cir/gpuprocessing/GpuMomentCounter.h"

namespace cir { namespace gpuprocessing {

class GpuImageProcessingService : public cir::common::ImageProcessingService {
public:
	GpuImageProcessingService(cir::common::logger::Logger& logger);
	virtual ~GpuImageProcessingService();
	void init(int width, int height);

	virtual const char* getModule();

	virtual cir::common::MatWrapper mark(cir::common::MatWrapper& input, cir::common::SegmentArray* segmentArray);
	virtual cir::common::MatWrapper crop(cir::common::MatWrapper& input, cir::common::Segment* segment);

protected:
	virtual cir::common::MatWrapper doToGrey(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper doThreshold(const cir::common::MatWrapper& input, double thresholdValue);
	virtual cir::common::MatWrapper doLowPass(const cir::common::MatWrapper& input, int size);
	virtual cir::common::MatWrapper doHighPass(const cir::common::MatWrapper& input, int size);
	virtual cir::common::MatWrapper doBgrToHsv(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper doHsvToBgr(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper doDetectColorHsv(const cir::common::MatWrapper& input,
			const int hsvRangesNumber, const cir::common::HsvRange* hsvRanges);
	virtual cir::common::MatWrapper doErode(const cir::common::MatWrapper& input, int times);
	virtual cir::common::MatWrapper doDilate(const cir::common::MatWrapper& input, int times);
	virtual cir::common::SegmentArray* doSegmentate(const cir::common::MatWrapper& input);
	virtual double* doCountHuMoments(const cir::common::MatWrapper& matWrapper);

private:
	GpuColorDetector _gpuColorDetector;
	GpuRegionSplittingSegmentator _segmentator;
	GpuMomentCounter _gpuMomentCounter;
};

}}
#endif
