#ifndef CPUBLURER_H_
#define CPUBLURER_H_

#include "cir/common/Blurer.h"

namespace cir { namespace cpuprocessing {

class CpuBlurer : public cir::common::Blurer {
public:
	CpuBlurer();
	virtual ~CpuBlurer();

protected:
	virtual cir::common::MatWrapper doMedian(const cir::common::MatWrapper& mw, int size);
};

}}
#endif /* CPUBLURER_H_ */
