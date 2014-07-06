#ifndef IMAGEPROCESSINGSERVICE_H_
#define IMAGEPROCESSINGSERVICE_H_

#include "cir/common/MatWrapper.h"

namespace cir { namespace common {

class ImageProcessingService {
public:
	ImageProcessingService();
	virtual ~ImageProcessingService();

	virtual MatWrapper toGrey(const MatWrapper& input) = 0;
	virtual MatWrapper threshold(const MatWrapper& input, double thresholdValue) = 0;
};

}}
#endif
