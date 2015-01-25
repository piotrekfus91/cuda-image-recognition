#ifndef BLURER_H_
#define BLURER_H_

#include "cir/common/MatWrapper.h"

namespace cir { namespace common {

class Blurer {
public:
	Blurer();
	virtual ~Blurer();

	virtual MatWrapper median(const MatWrapper& mw, int size = 2);

protected:
	virtual MatWrapper doMedian(const MatWrapper& mw, int size) = 0;
};

}}
#endif /* BLURER_H_ */
