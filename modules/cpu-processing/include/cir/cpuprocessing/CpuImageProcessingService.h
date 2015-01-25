#ifndef CPUIMAGEPROCESSINGSERVICE_H_
#define CPUIMAGEPROCESSINGSERVICE_H_

#include "cir/common/ImageProcessingService.h"
#include "cir/cpuprocessing/CpuColorDetector.h"
#include "cir/cpuprocessing/CpuRegionGrowingSegmentator.h"
#include "cir/cpuprocessing/CpuRegionSplittingSegmentator.h"
#include "cir/cpuprocessing/CpuRedMarker.h"
#include "cir/cpuprocessing/CpuMomentCounter.h"

namespace cir { namespace cpuprocessing {

class CpuImageProcessingService : public cir::common::ImageProcessingService {
public:
	CpuImageProcessingService(cir::common::logger::Logger& logger);
	virtual ~CpuImageProcessingService();
	void init(int width, int height);

	virtual const char* getModule();
	virtual void setSegmentator(cir::common::Segmentator* segmentator);
	virtual void setSegmentatorMinSize(int minSize);

	virtual cir::common::MatWrapper mark(cir::common::MatWrapper& input, const cir::common::SegmentArray* segmentArray);
	virtual cir::common::MatWrapper crop(cir::common::MatWrapper& input, cir::common::Segment* segment);

protected:
	virtual cir::common::MatWrapper doToGrey(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper doThreshold(const cir::common::MatWrapper& input, bool invertColors,
			double thresholdValue);
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

	virtual cir::common::MatWrapper getMatWrapper(const cv::Mat& mat) const;

private:
	CpuColorDetector _cpuColorDetector;
	cir::common::Segmentator* _segmentator;
	CpuRedMarker _marker;
	CpuMomentCounter _cpuMomentCounter;
};

}}
#endif
