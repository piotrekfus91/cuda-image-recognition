#ifndef GPUIMAGEPROCESSINGSERVICE_H_
#define GPUIMAGEPROCESSINGSERVICE_H_

#include "cir/common/ImageProcessingService.h"

namespace cir { namespace gpuprocessing {

class GpuImageProcessingService : public cir::common::ImageProcessingService {
public:
	GpuImageProcessingService();
	virtual ~GpuImageProcessingService();

	virtual cir::common::MatWrapper toGrey(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper threshold(const cir::common::MatWrapper& input, double thresholdValue);
};

}}
#endif
