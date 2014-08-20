#ifndef IMAGEPROCESSINGSERVICE_H_
#define IMAGEPROCESSINGSERVICE_H_

#include "cir/common/MatWrapper.h"
#include "cir/common/SegmentArray.h"

namespace cir { namespace common {

class ImageProcessingService {
protected:
	static int DEFAULT_LOW_PASS_KERNEL_SIZE;

public:
	ImageProcessingService();
	virtual ~ImageProcessingService();

	virtual MatWrapper toGrey(const MatWrapper& input) = 0;
	virtual MatWrapper threshold(const MatWrapper& input, double thresholdValue) = 0;
	virtual MatWrapper lowPass(const MatWrapper& input, int size = 3) = 0;
	virtual MatWrapper highPass(const MatWrapper& input, int size = 1) = 0;
	virtual MatWrapper bgrToHsv(const MatWrapper& input) = 0;
	virtual MatWrapper hsvToBgr(const MatWrapper& input) = 0;
	virtual MatWrapper detectColorHsv(const MatWrapper& input, const double minHue,
			const double maxHue, const double minSaturation, const double maxSaturation,
			const double minValue, const double maxValue) = 0;
	virtual SegmentArray* segmentate(const MatWrapper& input) = 0;
	virtual MatWrapper mark(MatWrapper& input, SegmentArray* segmentArray) = 0;

protected:
	static cv::Mat DEFAULT_LOW_PASS_KERNEL;
};

}}
#endif
