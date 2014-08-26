#ifndef CPUMOMENTCOUNTER_H_
#define CPUMOMENTCOUNTER_H_

#include "cir/common/MomentCounter.h"

namespace cir { namespace cpuprocessing {

class CpuMomentCounter : public cir::common::MomentCounter {
public:
	CpuMomentCounter();
	virtual ~CpuMomentCounter();

	virtual double* countHuMoments(cir::common::MatWrapper& matWrapper);
};

}}
#endif /* CPUMOMENTCOUNTER_H_ */
