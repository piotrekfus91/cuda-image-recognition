#ifndef GPUIMAGEPROCESSINGSERVICE_H_
#define GPUIMAGEPROCESSINGSERVICE_H_

#include "cir/common/ImageProcessingService.h"
#include "cir/common/Segmentator.h"
#include "cir/gpuprocessing/GpuColorDetector.h"
#include "cir/gpuprocessing/GpuRegionSplittingSegmentator.h"
#include "cir/gpuprocessing/GpuMomentCounter.h"
#include "cir/gpuprocessing/GpuBlurer.h"
#include "cir/gpuprocessing/GpuRedMarker.h"

namespace cir { namespace gpuprocessing {

class GpuImageProcessingService : public cir::common::ImageProcessingService {
public:
	GpuImageProcessingService(cir::common::logger::Logger& logger);
	virtual ~GpuImageProcessingService();
	void init(int width, int height);

	virtual const char* getModule();
	virtual void setSegmentatorMinSize(int minSize);
	virtual void setSegmentator(cir::common::Segmentator* segmentator);

	virtual cir::common::MatWrapper mark(cir::common::MatWrapper& input,
			const cir::common::SegmentArray* segmentArray);
	virtual cir::common::MatWrapper mark(cir::common::MatWrapper& input,
			std::vector<std::pair<cir::common::Segment*, int> > pairs);
	virtual cir::common::MatWrapper crop(cir::common::MatWrapper& input, cir::common::Segment* segment);

	virtual cir::common::MatWrapper getMatWrapper(const cv::Mat& mat) const;
	virtual cv::Mat getMat(const cir::common::MatWrapper& matWrapper) const;

protected:
	virtual cir::common::MatWrapper doToGrey(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper doThreshold(const cir::common::MatWrapper& input, bool invertColors,
			double thresholdValue);
	virtual cir::common::MatWrapper doLowPass(const cir::common::MatWrapper& input, int size);
	virtual cir::common::MatWrapper doMedian(const cir::common::MatWrapper& input, int size);
	virtual cir::common::MatWrapper doHighPass(const cir::common::MatWrapper& input, int size);
	virtual cir::common::MatWrapper doBgrToHsv(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper doHsvToBgr(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper doDetectColorHsv(const cir::common::MatWrapper& input,
			const int hsvRangesNumber, const cir::common::HsvRange* hsvRanges);
	virtual cir::common::MatWrapper doErode(const cir::common::MatWrapper& input, int times);
	virtual cir::common::MatWrapper doDilate(const cir::common::MatWrapper& input, int times);
	virtual cir::common::MatWrapper doEqualizeHistogram(const cir::common::MatWrapper& input);
	virtual cir::common::SegmentArray* doSegmentate(const cir::common::MatWrapper& input);
	virtual double* doCountHuMoments(const cir::common::MatWrapper& matWrapper);

private:
	GpuColorDetector _gpuColorDetector;
	GpuMomentCounter _gpuMomentCounter;
	GpuBlurer _blurer;
	GpuRedMarker _marker;
	cir::common::Segmentator* _segmentator;
};

}}
#endif
