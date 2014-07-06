#ifndef CPUIMAGEPROCESSINGSERVICE_H_
#define CPUIMAGEPROCESSINGSERVICE_H_

#include "cir/common/ImageProcessingService.h"

namespace cir { namespace cpuprocessing {

class CpuImageProcessingService : public cir::common::ImageProcessingService {
public:
	CpuImageProcessingService();
	virtual ~CpuImageProcessingService();

	virtual cir::common::MatWrapper toGrey(const cir::common::MatWrapper& input);
	virtual cir::common::MatWrapper threshold(const cir::common::MatWrapper& input, double thresholdValue);
};

}}
#endif
