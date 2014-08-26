#ifndef MOMENTCOUNTER_H_
#define MOMENTCOUNTER_H_

#include "cir/common/MatWrapper.h"

namespace cir { namespace common {

class MomentCounter {
public:
	MomentCounter();
	virtual ~MomentCounter();

	virtual double* countHuMoments(cir::common::MatWrapper& matWrapper) = 0;
};

}}
#endif /* MOMENTCOUNTER_H_ */
