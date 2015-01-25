#ifndef GPUBLURER_H_
#define GPUBLURER_H_

#include "cir/common/Blurer.h"

namespace cir { namespace gpuprocessing {

class GpuBlurer : public cir::common::Blurer {
public:
	GpuBlurer();
	virtual ~GpuBlurer();

protected:
	virtual cir::common::MatWrapper doMedian(const cir::common::MatWrapper& mw, int size);
};

}}
#endif /* GPUBLURER_H_ */
