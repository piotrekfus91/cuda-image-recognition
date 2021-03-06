#ifndef GPUMOMENTCOUNTER_H_
#define GPUMOMENTCOUNTER_H_

#include "cir/common/MatWrapper.h"

namespace cir { namespace gpuprocessing {

class GpuMomentCounter {
public:
	GpuMomentCounter();
	virtual ~GpuMomentCounter();
	virtual void init(int width, int height);

	virtual double* countHuMoments(cir::common::MatWrapper& matWrapper);
};

}}
#endif /* GPUMOMENTCOUNTER_H_ */
