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
	virtual cir::common::MatWrapper lowPass(const cir::common::MatWrapper& input, int size = DEFAULT_LOW_PASS_KERNEL_SIZE);
	virtual cir::common::MatWrapper highPass(const cir::common::MatWrapper& input, int size = 1);
};

}}
#endif
