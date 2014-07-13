#ifndef IMAGEPROCESSINGSERVICE_H_
#define IMAGEPROCESSINGSERVICE_H_

#include "cir/common/MatWrapper.h"

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

protected:
	static cv::Mat DEFAULT_LOW_PASS_KERNEL;
};

}}
#endif
